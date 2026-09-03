"""concurrency.py -- loop-local async primitives and a sync entry point that survives Jupyter.

Two hazards sit between this project's async I/O and the places it is called from, and both of them
fail in ways that look like something else entirely.

**1. asyncio primitives are bound to the loop they were first awaited on.** Since Python 3.10 a
``Semaphore`` or ``Lock`` raises ``RuntimeError: ... attached to a different loop`` the moment a
second event loop touches it. That is not hypothetical here: the reward callable handed to TRL is
invoked from inside the trainer, which runs it on a loop of its own, while the same
``AsyncPrimitives`` object is also used by conversation generation driven from the notebook's loop.
Caching one semaphore per object -- the obvious implementation -- therefore works for the first
phase of an iteration and then explodes partway through training, with a traceback that points at
the oracle call rather than at the cache. :class:`AsyncPrimitives` avoids it by creating each
primitive lazily, keyed by the running loop, and evicting entries belonging to loops that have
gone away. The key is ``id(loop)`` for the dict lookup, but the loop OBJECT is stored beside the
primitive and compared by identity on every hit: ``run_async`` creates and closes a fresh loop per
call, CPython reuses freed addresses eagerly, and an ``id``-only key would hand a primitive bound
to a dead loop to the new loop that inherited its address -- the same "attached to a different
loop" crash, now intermittent and address-dependent.

**2. Sync callers may already be inside a running loop.** Notebook cells and the trainers'
orchestration code are plain synchronous Python, but Jupyter owns a live event loop in the same
thread, so ``asyncio.run`` refuses. :func:`run_async` handles both cases: no loop running -> plain
``asyncio.run``; loop running -> a daemon thread with a fresh loop of its own, joined before
returning. Deliberately **not** ``nest_asyncio``, which is broken on Python >= 3.13 under the
stricter ``contextvars`` re-entry rules.

**The invariant everything else rests on: ``gpu_lock`` is held ONLY across a therapist
``generate``, NEVER across a patient ``await``.** The lock exists because concurrent
``model.generate`` calls on one CUDA context interleave badly, not because the pipeline is
serial -- so if it is ever held while awaiting an oracle or patient response, every simulation in
the batch queues behind that one network round-trip and the look-ahead's whole reason for being
batched disappears. Acquire it, generate, release it, then await.

Usage::

    from core.concurrency import AsyncPrimitives, run_async

    primitives = AsyncPrimitives(oracle_concurrency=64, patient_concurrency=96)
    states = run_async(generate_all_conversations(..., primitives=primitives, ...))
"""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import Future
from typing import Any, Callable, Coroutine, Dict, List, Tuple, TypeVar

__all__ = ["AsyncPrimitives", "run_async"]

_T = TypeVar("_T")

_ORACLE_SEM = "oracle_sem"
_PATIENT_SEM = "patient_sem"
_GPU_LOCK = "gpu_lock"


# ==============================================================================
#                        LOOP-LOCAL ASYNC PRIMITIVES
# ==============================================================================


class AsyncPrimitives:
    """Semaphores and the GPU lock, created lazily and keyed by the running event loop.

    One instance is built per trainer process and passed down through conversation generation,
    look-ahead and oracle scoring. It holds no primitives at construction time -- each accessor
    creates its primitive on first use *inside the calling loop*, caches it under
    ``id(that loop)`` together with the loop object itself, and drops any entry whose stored
    loop is not (by identity) the loop now running -- a reused address never resurrects a
    primitive bound to a closed loop.

    Args:
        oracle_concurrency: maximum in-flight oracle scoring calls.
        patient_concurrency: maximum in-flight patient calls.

    Notes:
        **One patient semaphore serves both callers.** Exp3 had a separate
        ``lookahead_patient_sem`` because the conversation loop and the look-ahead rollout ran
        against separately-provisioned OpenAI capacity. In Exp4 both hit the same local vLLM
        server, so a single bound is the honest one: two independent semaphores would let the
        two paths jointly exceed what the server can queue, which shows up as timeouts rather
        than as a queue.

        **Eviction assumes one live loop at a time.** :func:`run_async` joins its worker thread
        before returning, so loops are sequential in practice and evicting the previous loop's
        entry never discards a primitive anyone still holds. If two loops were ever genuinely
        concurrent they would evict each other on every access, each getting a fresh semaphore
        and silently losing the concurrency bound -- correctness is preserved (the primitives
        stay loop-local) but the limit stops limiting. Do not drive one instance from two
        simultaneously-running loops.
    """

    def __init__(self, *, oracle_concurrency: int, patient_concurrency: int) -> None:
        if oracle_concurrency < 1 or patient_concurrency < 1:
            raise ValueError(
                f"concurrency limits must be >= 1 "
                f"(got oracle={oracle_concurrency}, patient={patient_concurrency})"
            )
        self._oracle_concurrency = int(oracle_concurrency)
        self._patient_concurrency = int(patient_concurrency)
        # (name, id(loop)) -> (loop, primitive). The loop object is kept so a hit can be
        # verified by identity, not just by address (see the module docstring).
        self._cache: Dict[tuple, Tuple[asyncio.AbstractEventLoop, Any]] = {}

    # -- accessors -------------------------------------------------------------

    def oracle_sem(self) -> asyncio.Semaphore:
        """Bound on concurrent oracle scoring calls, local to the running loop."""
        return self._get(_ORACLE_SEM, lambda: asyncio.Semaphore(self._oracle_concurrency))

    def patient_sem(self) -> asyncio.Semaphore:
        """Bound on concurrent patient calls -- shared by the conversation loop and look-ahead."""
        return self._get(_PATIENT_SEM, lambda: asyncio.Semaphore(self._patient_concurrency))

    def gpu_lock(self) -> asyncio.Lock:
        """Serializes therapist ``model.generate`` calls on the single CUDA context.

        Hold it across the generate and nothing else. Holding it across an ``await`` on a
        patient or oracle call serializes the whole batch behind one network round-trip.
        """
        return self._get(_GPU_LOCK, asyncio.Lock)

    # -- diagnostics -----------------------------------------------------------

    def cached_loop_ids(self) -> Dict[str, List[int]]:
        """``{primitive name: [loop ids currently cached]}`` -- for smoke tests and debugging only.

        Under the single-live-loop invariant every list holds at most one id. A name with two ids
        means two loops touched this instance without either being evicted, i.e. they overlapped.
        """
        out: Dict[str, List[int]] = {}
        for name, loop_id in self._cache:
            out.setdefault(name, []).append(loop_id)
        return out

    # -- internals -------------------------------------------------------------

    def _get(self, name: str, factory: Callable[[], Any]) -> Any:
        """Fetch (or create) ``name`` for the currently-running loop, evicting stale loops' copies.

        Raises ``RuntimeError`` when called with no loop running -- the same error asyncio itself
        would raise, and a sign the caller forgot :func:`run_async`.

        A cached entry is a hit only when its stored loop IS the running loop. An entry whose
        address matches but whose object does not (a closed loop's address reused by a new one)
        is stale and is evicted like any other loop's entry.
        """
        loop = asyncio.get_running_loop()
        key = (name, id(loop))
        hit = self._cache.get(key)
        if hit is not None and hit[0] is loop:
            return hit[1]
        stale = [k for k in self._cache if k[0] == name]
        for k in stale:
            del self._cache[k]
        primitive = factory()
        self._cache[key] = (loop, primitive)
        return primitive


# ==============================================================================
#                          SYNC ENTRY POINT
# ==============================================================================


def run_async(coro: Coroutine[Any, Any, _T]) -> _T:
    """Run *coro* to completion from synchronous code, inside or outside a live event loop.

    With no loop running (plain scripts, ``python -m``) this is ``asyncio.run``. With a loop
    already running -- the Jupyter case, where the kernel owns the thread's loop and
    ``asyncio.run`` refuses -- the coroutine is executed on a **daemon thread with its own fresh
    loop**, which is then joined, so the call is still synchronous from the caller's point of view
    and exceptions propagate normally.

    Args:
        coro: the coroutine to run. It is consumed either way; a coroutine cannot be re-run, so
            build a new one per call.

    Returns:
        Whatever the coroutine returns.

    Notes:
        This is deliberately not ``nest_asyncio``: patching a running loop to be re-entrant is
        broken on Python >= 3.13 under the stricter ``contextvars`` re-entry rules, and the two
        trainers must run on 3.13.

        **Torch calls from that thread are fine.** CUDA context and PyTorch's allocator are
        per-PROCESS, not per-thread, so a coroutine doing ``model.generate`` on the worker thread
        sees the same device state as the notebook. This is why the whole look-ahead rollout can
        run under ``run_async`` from a notebook cell.

        The thread is a daemon so an interpreter shutdown while a call is wedged does not hang the
        process -- but the join here means normal control flow always waits for it.

        The worker loop is torn down the way ``asyncio.run`` tears its own down: async generators
        are finalised (``shutdown_asyncgens``) and the default executor is drained
        (``shutdown_default_executor``, Python >= 3.9) BEFORE ``close()``. The look-ahead runs
        its therapist generate through ``run_in_executor``, so closing without the drain would
        leave that worker thread orphaned on the dead loop and abandon whatever it was holding.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result_future: Future = Future()

    def _thread_target() -> None:
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result_future.set_result(loop.run_until_complete(coro))
            except BaseException as exc:                     # includes CancelledError / KeyboardInterrupt
                result_future.set_exception(exc)
            finally:
                try:
                    loop.run_until_complete(loop.shutdown_asyncgens())
                    loop.run_until_complete(loop.shutdown_default_executor())
                finally:
                    asyncio.set_event_loop(None)
                    loop.close()
        except BaseException as exc:                         # loop creation or teardown failed
            if not result_future.done():
                result_future.set_exception(exc)

    t = threading.Thread(target=_thread_target, daemon=True)
    t.start()
    t.join()
    return result_future.result()
