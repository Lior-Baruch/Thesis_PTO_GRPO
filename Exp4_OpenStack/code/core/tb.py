"""tb.py -- TensorBoard is the ONLY telemetry Exp4 has.

Exp3 logged to Weights and Biases *and* TensorBoard, and read most of its curves back out of
W&B. Exp4 drops W&B entirely (one less account, one less network dependency, one less thing that
can stall a Colab cell mid-iteration), which promotes TensorBoard from "the mirror" to "the
record". Everything a run knows about itself while it is running passes through this module.

Three concerns share the file, in the order a run touches them:

1. **Trainer wiring** -- :func:`setup_tensorboard_logging` and
   :func:`patch_trainer_tensorboard_callback` force HuggingFace's ``TensorBoardCallback`` to write
   where we say. Without them the callback invents its own directory name from the clock and the
   hostname, which is how Exp3 collected ``WinError 123`` (invalid filename) on Windows. Exp4 always
   passes an explicit ASCII logdir, and refuses one that is not a legal path segment everywhere the
   run tree travels -- Colab writes it, Google Drive syncs it, Windows reads it.

2. **The live run-level view** -- :class:`RunTBLogger`. Each iteration builds a *fresh* Trainer, so
   TRL's own event files restart ``global_step`` at 0 and the TB web UI cannot draw a continuous
   cross-iteration curve. This writer sits outside the trainers, at ``runs/<ARM>/tb_live/``, and logs
   at the cumulative step. It is **opt-in** (``enabled=False`` by default) and it is entirely
   optional: every backend call is swallowed, because telemetry must never be able to kill an
   iteration that costs GPU-hours.

3. **The post-hoc dashboard** -- :func:`parse_tensorboard_logs` and friends. The same
   restart-at-zero problem, solved on the read side: each event file's steps are re-based by the
   ``iteration_N`` component of its path, so iteration 2's step 50 lands after iteration 1's step 50
   instead of colliding with it on dedup. This is what makes a cross-iteration curve possible at all
   from the per-iteration event files.

Everything heavy (pandas, matplotlib, tensorboard, torch) is imported INSIDE functions. The trainer
imports this module on a Colab worker where the plotting stack is irrelevant, and the reader side
runs locally without touching torch; neither should pay for the other.
"""

from __future__ import annotations

import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "TB_LOGDIR_ENV",
    "EVENT_FILE_GLOB",
    "LIVE_SUBDIR",
    "setup_tensorboard_logging",
    "patch_trainer_tensorboard_callback",
    "RunTBLogger",
    "find_event_files",
    "parse_tensorboard_logs",
    "scan_scalar_tags",
    "compute_iteration_boundaries",
    "plot_iteration_metrics",
]


#: The environment variable ``transformers.TensorBoardCallback`` reads -- in ``__init__``, not at
#: train time, which is why :func:`setup_tensorboard_logging` must run BEFORE the Trainer is built.
TB_LOGDIR_ENV = "TENSORBOARD_LOGGING_DIR"

#: Filename pattern every TB writer produces.
EVENT_FILE_GLOB = "events.out.tfevents.*"

#: The run-level live view's subdirectory. Its steps are ALREADY cumulative, so the post-hoc parser
#: excludes it by default -- mixing it into the per-iteration re-basing would double-count.
LIVE_SUBDIR = "tb_live"

_ITER_DIR_RE = re.compile(r"iteration_(\d+)")

#: Illegal in a Windows path segment. The run tree is written on Colab and read on Windows through
#: Google Drive, so a name that is merely POSIX-legal is not good enough.
_WINDOWS_ILLEGAL = set('<>:"|?*') | {chr(c) for c in range(32)}

_TIDY_COLUMNS = ("iteration", "step", "global_step", "tag", "value", "wall_time", "event_file")


# ==============================================================================
# TRAINER-SIDE SETUP
# ==============================================================================


def _check_logdir_name(path: str) -> None:
    """Raise if any segment of *path* would be illegal on NTFS; warn on non-ASCII.

    The drive letter's colon is skipped (``C:``), everything after it is checked. Arm names come
    from ``naming.build_experiment_name`` and are ``[A-Za-z0-9_]`` by construction, so this can only
    fire on a hand-typed prefix -- which is exactly when a clear error beats ``WinError 123``.
    """
    _, tail = os.path.splitdrive(path)
    for part in Path(tail).parts:
        if part in ("/", "\\", os.sep):
            continue
        bad = sorted(_WINDOWS_ILLEGAL & set(part))
        if bad:
            raise ValueError(
                f"TensorBoard logdir segment {part!r} contains characters that are illegal in a "
                f"Windows path ({bad}). The run tree is synced to Windows via Drive; pick an "
                f"ASCII, alphanumeric name."
            )
        if any(ord(ch) > 127 for ch in part):
            print(f"  [tb] WARNING: logdir segment {part!r} is not ASCII; expect console mojibake.")


def setup_tensorboard_logging(tensorboard_log_dir: str) -> None:
    """Point HuggingFace's ``TensorBoardCallback`` at an explicit, Windows-safe directory.

    Args:
        tensorboard_log_dir: where this iteration's event files go, typically
            ``<run_dir>/iteration_<N>/training/tb_logs``. Created if absent, and stored as an
            ABSOLUTE path so a later ``os.chdir`` (the trainer notebooks cd into ``code/`` after
            mounting Drive) cannot re-point it.

    Notes:
        **Call this BEFORE constructing the Trainer.** ``TensorBoardCallback.__init__`` reads
        ``TENSORBOARD_LOGGING_DIR`` once and keeps the value; setting the variable afterwards has no
        effect, and :func:`patch_trainer_tensorboard_callback` exists for exactly that case.

        Why an explicit directory at all: when ``logging_dir`` is ``None`` the callback falls back
        to ``default_logdir()``, an implicit ``runs/<Mon><DD>_<HH-MM-SS>_<hostname>`` built from the
        clock and the machine name. That name is neither stable nor guaranteed to be a legal
        Windows path segment -- Exp3 hit ``WinError 123`` from it -- and it scatters event files
        outside the iteration tree, which breaks the path-based re-basing in
        :func:`parse_tensorboard_logs`.
    """
    if not tensorboard_log_dir or not str(tensorboard_log_dir).strip():
        raise ValueError("tensorboard_log_dir must be a non-empty path")
    log_dir = os.path.abspath(str(tensorboard_log_dir))
    _check_logdir_name(log_dir)
    os.makedirs(log_dir, exist_ok=True)
    os.environ[TB_LOGDIR_ENV] = log_dir
    print(f"  [tb] log dir: {log_dir}")


def patch_trainer_tensorboard_callback(trainer: Any, tensorboard_log_dir: str) -> None:
    """Re-point a Trainer's already-constructed ``TensorBoardCallback`` at *tensorboard_log_dir*.

    Args:
        trainer: an HF/TRL trainer exposing ``callback_handler.callbacks``.
        tensorboard_log_dir: the same directory handed to :func:`setup_tensorboard_logging`.

    Notes:
        Needed because the callback may already exist by the time we get a say -- TRL constructs the
        callback list inside ``Trainer.__init__``, and a resumed iteration reuses a callback whose
        ``logging_dir`` was captured from a previous environment. Setting ``cb.logging_dir`` alone
        is not enough: if the writer is already open it keeps writing to the old directory, so the
        writer is closed and **nulled** and the callback reopens it lazily against the new path.

        Matched on class NAME rather than ``isinstance`` so this file need not import transformers
        (heavy imports stay lazy) and so a tensorboardX-backed subclass is caught too. A missing
        callback is a warning, not an exception -- logging must not be able to abort a run.
    """
    log_dir = os.path.abspath(str(tensorboard_log_dir))
    _check_logdir_name(log_dir)
    os.makedirs(log_dir, exist_ok=True)
    # Insurance for any trainer constructed later in the same process.
    os.environ[TB_LOGDIR_ENV] = log_dir

    handler = getattr(trainer, "callback_handler", None)
    callbacks = list(getattr(handler, "callbacks", []) or [])
    patched = 0
    for cb in callbacks:
        if cb.__class__.__name__ != "TensorBoardCallback":
            continue
        cb.logging_dir = log_dir
        writer = getattr(cb, "tb_writer", None)
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass
            cb.tb_writer = None
        patched += 1
    if patched:
        print(f"  [tb] patched {patched} TensorBoardCallback(s) -> {log_dir}")
    else:
        print("  [tb] WARNING: no TensorBoardCallback on this trainer; is report_to set?")


# ==============================================================================
# LIVE RUN-LEVEL LOGGER (opt-in)
# ==============================================================================


def _finite(v: Any) -> bool:
    try:
        return math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def _fmt(v: Any) -> str:
    """Format a score-ish value for display; ``'-'`` for None/NaN."""
    if v is None or not _finite(v):
        return "-"
    return f"{float(v):.3f}"


def _clip(s: Any, n: int) -> str:
    s = "" if s is None else str(s)
    return s if len(s) <= n else s[:n] + "..."


def _tail_str(s: Any, n: int) -> str:
    s = "" if s is None else str(s)
    return s if len(s) <= n else "..." + s[-n:]


def _format_samples_markdown(samples: Sequence[dict]) -> str:
    """Render candidate records as TB-friendly markdown.

    Each record is the recorder's shape: ``score``, ``sub_scores``, ``completion``, ``prompt``, and
    optionally ``lookahead`` / ``pto`` / ``grpo`` context blocks. Unknown keys are ignored, so this
    survives the recorder growing fields.
    """
    blocks: List[str] = []
    for i, r in enumerate(samples):
        sub = r.get("sub_scores") or {}
        sub_s = ", ".join(f"Q{k}={_fmt(v)}" for k, v in sub.items())
        if r.get("pto"):
            ctx = f"role=**{(r['pto'] or {}).get('role')}**"
        elif r.get("grpo"):
            g = r["grpo"] or {}
            ctx = f"group_mean={_fmt(g.get('group_mean'))} group_std={_fmt(g.get('group_std'))}"
        else:
            ctx = ""
        la = r.get("lookahead") or {}
        la_s = f" - lookahead {la.get('realized_turns')}/{la.get('k')}" if la.get("k") else ""
        blocks.append(
            f"**#{i} -- score {_fmt(r.get('score'))}** {ctx}"
            f"{(' (' + sub_s + ')') if sub_s else ''}{la_s}\n\n"
            f"_prompt tail:_ `{_tail_str(r.get('prompt'), 280)}`\n\n"
            f"_completion:_ {_clip(r.get('completion'), 800)}\n\n---"
        )
    return "\n\n".join(blocks)


#: Panes for ``add_custom_scalars``. Purely a TB-UI grouping: it does not create tags, it arranges
#: the ones the trainers happen to write.
_CUSTOM_LAYOUT: Dict[str, Dict[str, list]] = {
    "EDA": {
        "candidate_reward": ["Multiline", ["eda/mean_candidate_reward"]],
        "oracle_success_rate": ["Multiline", ["eda/oracle_success_rate"]],
        "lookahead": ["Multiline", ["eda/lookahead_realized_turns_mean",
                                    "eda/lookahead_ended_early_frac"]],
    },
    "PTO": {
        "pref_pairs": ["Multiline", ["pto/pref_pair_count", "pto/branch_points"]],
        "tau_filter_rate": ["Multiline", ["pto/tau_filter_rate"]],
    },
    "GRPO": {
        "group_std": ["Multiline", ["grpo/group_reward_std_mean"]],
        "frac_zero_std": ["Multiline", ["grpo/frac_zero_std"]],
    },
}


class RunTBLogger:
    """The continuous run-level TensorBoard view at ``<run_dir>/tb_live/``.

    TRL writes one event file per iteration, each starting at ``global_step`` 0, so the TB web UI
    shows N disconnected curves; the matplotlib dashboard in this module stitches them, but only
    after the fact. This writer is the *during-training* answer: ONE ``SummaryWriter`` for the whole
    run, logged at the cumulative step, so the web UI renders one smoothable curve per tag while the
    run is still going. It carries the aggregates TRL does not know about -- mean candidate reward,
    oracle success rate, look-ahead realized turns, PTO's tau-filter rate, GRPO's group std -- plus
    per-iteration reward histograms and a readable sample of completions.

    Args:
        run_dir: the arm's run directory (``data/runs/<EXP_NAME>``). ``tb_live/`` is created inside.
        enabled: **default False** -- the live view is opt-in. When False every method is a no-op
            and no writer is ever opened, so the class costs nothing to construct unconditionally.

    Notes:
        This writer is a convenience, never a source of truth: the per-iteration event files TRL
        writes are the record. Accordingly EVERY backend call is wrapped -- a full disk, a Drive
        stall, or a torch/tensorboard version skew degrades this object to a no-op rather than
        raising into an optimizer step that cost GPU-hours to reach. Repeated failures of the same
        operation print once and are counted, so a per-step failure cannot flood the Colab log.

        ``tb_live/`` is deliberately EXCLUDED from :func:`find_event_files` by default: its steps
        are already cumulative, and re-basing them alongside the per-iteration files would
        double-count.
    """

    def __init__(self, run_dir: str, *, enabled: bool = False) -> None:
        self.enabled = bool(enabled)
        self.run_dir = str(run_dir)
        self.log_dir = os.path.join(self.run_dir, LIVE_SUBDIR)
        self.writer = None
        self._layout_written = False
        self._warn_counts: Dict[str, int] = {}
        if not self.enabled:
            return
        try:
            from torch.utils.tensorboard import SummaryWriter  # lazy: torch is trainer-side only

            os.makedirs(self.log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=self.log_dir)
            print(f"  [tb] live run-level logger -> {self.log_dir}")
        except Exception as e:
            print(f"  [tb] WARNING: SummaryWriter unavailable ({e}); live view disabled")
            self.enabled = False
            self.writer = None

    # -- internals ---------------------------------------------------------

    @property
    def _live(self) -> bool:
        return bool(self.enabled and self.writer is not None)

    def _warn(self, key: str, message: str) -> None:
        """Print the first failure of each kind, then just count the rest."""
        n = self._warn_counts.get(key, 0) + 1
        self._warn_counts[key] = n
        if n == 1:
            print(f"  [tb] WARNING: {message}")

    def _write_custom_layout(self) -> None:
        """Emit the EDA / PTO / GRPO pane grouping. Once per writer -- torch warns on a second call."""
        if not self._live or self._layout_written:
            return
        self._layout_written = True
        try:
            self.writer.add_custom_scalars(_CUSTOM_LAYOUT)
        except Exception as e:
            self._warn("layout", f"add_custom_scalars failed ({e})")

    def _flush(self) -> None:
        try:
            self.writer.flush()
        except Exception as e:
            self._warn("flush", f"flush failed ({e})")

    # -- public surface ----------------------------------------------------

    def log_scalars(self, scalars: Dict[str, float], *, step: int, iteration: int) -> None:
        """Log a batch of run-level scalars at the CUMULATIVE step.

        Args:
            scalars: ``{tag: value}``. ``None`` and non-finite values are dropped rather than
                written -- a NaN in a TB scalar poisons the whole curve's autoscale.
            step: cumulative global step across iterations (see
                ``policy.compute_cumulative_step_offset``). Passing a per-iteration step here would
                overwrite earlier iterations' points, which is the exact failure this class exists
                to avoid.
            iteration: the iteration these scalars came from. Mirrored as the scalar
                ``run/iteration`` so the live view can be read back as "which iteration was step X".
        """
        if not self._live:
            return
        self._write_custom_layout()
        clean = {k: float(v) for k, v in (scalars or {}).items() if v is not None and _finite(v)}
        clean.setdefault("run/iteration", float(iteration))
        for k, v in clean.items():
            try:
                self.writer.add_scalar(k, v, global_step=int(step))
            except Exception as e:
                self._warn(f"scalar:{k}", f"add_scalar({k}) failed ({e})")
        self._flush()

    def log_histogram(self, tag: str, values: Any, *, step: int, iteration: int) -> None:
        """Log a distribution (typically an iteration's candidate rewards) at the cumulative step.

        Non-finite entries are filtered out; an all-empty input is a silent no-op. ``iteration`` is
        recorded alongside so a histogram can be attributed without consulting the boundaries.
        """
        if not self._live:
            return
        try:
            import numpy as np  # lazy: pulled in by tensorboard anyway, but not at import time

            arr = np.asarray([float(v) for v in (values or []) if _finite(v)], dtype=float)
        except Exception as e:
            self._warn("hist-prep", f"histogram values unusable ({e})")
            return
        if arr.size == 0:
            return
        try:
            self.writer.add_histogram(tag, arr, global_step=int(step))
            self.writer.add_scalar("run/iteration", float(iteration), global_step=int(step))
            self._flush()
        except Exception as e:
            self._warn("hist", f"add_histogram({tag}) failed ({e})")

    def log_text(self, tag: str, text: str, *, step: int) -> None:
        """Log a markdown blob (TB renders it in the TEXT tab)."""
        if not self._live:
            return
        try:
            self.writer.add_text(tag, str(text), global_step=int(step))
            self._flush()
        except Exception as e:
            self._warn("text", f"add_text({tag}) failed ({e})")

    def log_sample_completions(self, samples: List[dict], *, step: int, iteration: int) -> None:
        """Log a readable spread of candidate records (best / median / worst) as TB markdown.

        Replaces dumping completions to stdout: the same information, browsable in the TB UI and
        attributable to an iteration, without megabytes of Colab scrollback. The caller chooses the
        spread; this method just formats what it is given.
        """
        if not self._live or not samples:
            return
        self.log_text(f"samples/iteration_{iteration}", _format_samples_markdown(samples), step=step)

    def close(self) -> None:
        """Flush and close. Idempotent; safe to call on a disabled logger."""
        if self.writer is not None:
            try:
                self.writer.flush()
                self.writer.close()
            except Exception:
                pass
            self.writer = None
        if self._warn_counts:
            total = sum(self._warn_counts.values())
            print(f"  [tb] live logger closed with {total} suppressed warning(s): "
                  f"{sorted(self._warn_counts)}")


# ==============================================================================
# EVENT-FILE PARSING (post-hoc)
# ==============================================================================


def _iteration_from_path(path: str) -> int:
    """Iteration number from an ``.../iteration_N/...`` path; 0 when there is no such segment.

    The path is absolutised first so pointing the reader straight at one iteration's ``tb_logs``
    still recovers N from the parent directories.
    """
    for part in Path(os.path.abspath(path)).parts:
        m = _ITER_DIR_RE.fullmatch(part)
        if m:
            return int(m.group(1))
    return 0


def find_event_files(log_root: str, *, include_live: bool = False) -> List[str]:
    """Every ``events.out.tfevents.*`` under *log_root*, ordered by (iteration, path).

    Args:
        log_root: directory to recurse. A missing directory returns ``[]`` rather than raising --
            the dashboard is routinely pointed at a run that has not trained yet.
        include_live: include files under ``tb_live/``. **Default False on purpose**: the live
            view's steps are already cumulative, so re-basing them next to the per-iteration files
            would double-count every point.

    Notes:
        Sorted NUMERICALLY by iteration, not lexically -- ``iteration_10`` sorts before
        ``iteration_2`` as a string, and the step-offset accumulation in
        :func:`parse_tensorboard_logs` depends on the true order.
    """
    root = os.path.abspath(str(log_root))
    if not os.path.isdir(root):
        return []
    found: List[Tuple[int, str, str]] = []
    for p in Path(root).rglob(EVENT_FILE_GLOB):
        sp = str(p)
        if not include_live and LIVE_SUBDIR in p.parts:
            continue
        found.append((_iteration_from_path(sp), sp, sp))
    found.sort(key=lambda t: (t[0], t[1]))
    return [f[2] for f in found]


def _empty_tidy_frame():
    import pandas as pd

    return pd.DataFrame({c: [] for c in _TIDY_COLUMNS})


def parse_tensorboard_logs(log_root: str, *, include_live: bool = False) -> "Any":
    """Read every scalar under *log_root* into one tidy, step-re-based DataFrame.

    Returns:
        A DataFrame with columns ``iteration, step, global_step, tag, value, wall_time,
        event_file``. ``step`` is the raw within-iteration step TRL wrote; ``global_step`` is that
        step shifted by the total steps completed in EARLIER iterations. An empty frame with the
        same columns is returned when nothing is found, so callers can branch on ``.empty``.

    Notes:
        **The re-basing is the point.** Each iteration builds a fresh Trainer, so every event file
        starts at step 0; without the shift, iteration 1 step 50 and iteration 2 step 50 are the
        same (tag, step) pair and one of them is dropped by dedup -- the curve silently loses data
        instead of chaining end to end. The offset for iteration N is the sum of the max step
        observed in iterations 1..N-1.

        A resumed iteration writes a SECOND event file into the same ``tb_logs`` directory with
        overlapping steps. Both are read; duplicates on ``(tag, global_step)`` resolve to the
        LATEST-written value by ``wall_time``, i.e. the resumed run wins over the crashed one.

        Unreadable event files are warned about and skipped -- a torn file from a killed Colab
        session must not take the whole dashboard down.
    """
    import pandas as pd
    from tensorboard.backend.event_processing import event_accumulator as ea

    event_files = find_event_files(log_root, include_live=include_live)
    if not event_files:
        return _empty_tidy_frame()

    rows: List[dict] = []
    per_iter_max: Dict[int, int] = {}
    for fp in event_files:
        it = _iteration_from_path(fp)
        try:
            acc = ea.EventAccumulator(fp, size_guidance={ea.SCALARS: 0})
            acc.Reload()
            tags = acc.Tags().get("scalars", [])
        except Exception as e:
            print(f"  [tb] WARNING: could not read {fp} ({e})")
            continue
        for tag in tags:
            try:
                events = acc.Scalars(tag)
            except Exception as e:
                print(f"  [tb] WARNING: could not read tag {tag!r} in {fp} ({e})")
                continue
            for ev in events:
                rows.append({
                    "iteration": it,
                    "step": int(ev.step),
                    "global_step": int(ev.step),          # re-based below
                    "tag": tag,
                    "value": float(ev.value),
                    "wall_time": float(ev.wall_time),
                    "event_file": fp,
                })
                if ev.step > per_iter_max.get(it, 0):
                    per_iter_max[it] = int(ev.step)

    if not rows:
        return _empty_tidy_frame()

    offsets: Dict[int, int] = {}
    cumulative = 0
    for it in sorted(per_iter_max):
        offsets[it] = cumulative
        cumulative += per_iter_max[it]

    df = pd.DataFrame(rows)
    df["global_step"] = df["step"] + df["iteration"].map(offsets).fillna(0).astype(int)
    df = (df.sort_values(["tag", "global_step", "wall_time"])
            .drop_duplicates(subset=["tag", "global_step"], keep="last")
            .sort_values(["global_step", "tag"])
            .reset_index(drop=True))
    return df[list(_TIDY_COLUMNS)]


def scan_scalar_tags(log_root: str, *, include_live: bool = False) -> List[str]:
    """Every scalar tag present under *log_root*, sorted and deduplicated.

    Cheap by design: it reads tag names only (``size_guidance`` of 1 scalar per tag), so it is the
    right call for "what did this run actually log?" before pointing the dashboard at it.

    Returns ``[]`` -- never raises -- for a missing directory, an empty run, or a corrupt event
    file, because the usual reason to call it is that something looks wrong already.
    """
    from tensorboard.backend.event_processing import event_accumulator as ea

    event_files = find_event_files(log_root, include_live=include_live)
    if not event_files:
        print(f"  [tb] no event files under {os.path.abspath(str(log_root))}")
        return []

    tags: set = set()
    for fp in event_files:
        try:
            acc = ea.EventAccumulator(fp, size_guidance={ea.SCALARS: 1})
            acc.Reload()
            tags.update(acc.Tags().get("scalars", []))
        except Exception as e:
            print(f"  [tb] WARNING: could not inspect {fp} ({e})")
            continue
    out = sorted(tags)
    print(f"  [tb] {len(event_files)} event file(s), {len(out)} unique scalar tag(s)")
    return out


def compute_iteration_boundaries(df: Any) -> List[int]:
    """Cumulative step at the END of each iteration, ascending.

    Args:
        df: the tidy frame from :func:`parse_tensorboard_logs`. A path string is also accepted as a
            notebook convenience and is parsed first.

    Returns:
        One ``global_step`` per iteration, in iteration order. The LAST entry is the right edge of
        the plot, not an internal boundary -- :func:`plot_iteration_metrics` skips it when drawing
        separators. Returns ``[]`` for an empty or malformed frame.
    """
    if isinstance(df, (str, os.PathLike)):
        df = parse_tensorboard_logs(str(df))
    if df is None or getattr(df, "empty", True):
        return []
    if not {"iteration", "global_step"}.issubset(set(df.columns)):
        return []
    grouped = df.groupby("iteration")["global_step"].max().sort_index()
    return [int(v) for v in grouped.tolist()]


# ==============================================================================
# POST-HOC DASHBOARD
# ==============================================================================


def _split_tag(tag: str) -> Tuple[str, str]:
    """``'train/rewards/chosen'`` -> ``('train', 'rewards/chosen')``; unprefixed -> ``('', tag)``."""
    head, sep, rest = tag.partition("/")
    if sep and head in ("train", "eval"):
        return head, rest
    return "", tag


def _resolve_tag(tags: set, preferred: str) -> Optional[str]:
    """Best available tag for *preferred*: exact match, else the same body under the same split.

    TRL/HF prefix their scalars (``train/loss``, ``eval/loss``) and have moved the middle of the
    path between versions, so a pane spec must tolerate a rename -- but only in ways that cannot
    change what the curve MEANS.

    The rules, in order:

    1. exact hit;
    2. a tag with the same ``train``/``eval`` split whose body equals, or ends with ``/`` + , the
       requested body (so ``rewards/chosen`` can be found under a deeper prefix);
    3. for a ``train/`` request only, an unprefixed tag by the same rule (a bare tag is
       conventionally the training one).

    What it deliberately will NOT do is fall back to a bare leaf or a substring. ``eval/reward``
    must never resolve to ``train/reward`` (the pane would plot the train curve under an "Eval"
    label) and ``rewards/chosen`` must never resolve to ``logps/chosen`` (same leaf, different
    quantity). A missing tag drops its series; a mislabelled one is a wrong figure.

    Ties break on (length, name), so the most direct match wins and the result is deterministic.
    """
    if preferred in tags:
        return preferred
    want_head, want_body = _split_tag(preferred)
    allowed_heads = {want_head, ""} if want_head == "train" else {want_head}
    hits = []
    for t in tags:
        head, body = _split_tag(t)
        if head not in allowed_heads:
            continue
        if body == want_body or body.endswith("/" + want_body):
            hits.append(t)
    if not hits:
        return None
    return sorted(hits, key=lambda t: (len(t), t))[0]


def _smooth(xs: List[float], ys: List[float], window: int) -> Tuple[List[float], List[float]]:
    """Trailing moving average. Visualization only -- never feed this to an analysis."""
    if window <= 1 or len(ys) < window:
        return xs, ys
    out: List[float] = []
    run = 0.0
    for i, y in enumerate(ys):
        run += y
        if i >= window:
            run -= ys[i - window]
        out.append(run / min(i + 1, window))
    return xs, out


def _detect_method(tags: set, method: Optional[str]) -> str:
    """``'grpo'`` | ``'dpo'`` | ``'unknown'`` from an explicit hint or from the tags themselves.

    ``method='pto'`` maps to ``'dpo'``: PTO is the framework, DPO is the loss, and it is the loss
    that decides which scalars TRL wrote.
    """
    if method:
        m = str(method).strip().lower()
        if m in ("pto", "dpo"):
            return "dpo"
        if m == "grpo":
            return "grpo"
        print(f"  [tb] WARNING: unknown method {method!r}; auto-detecting instead")
    if any("rewards/chosen" in t for t in tags):
        return "dpo"
    if any(("reward_std" in t) or ("frac_reward_zero_std" in t) for t in tags):
        return "grpo"
    return "unknown"


# (title, ylabel, [(tag, label, color, linestyle), ...])
_PaneSpec = Tuple[str, str, List[Tuple[str, str, str, str]]]

_LOSS_PANE: _PaneSpec = ("Loss", "Loss", [
    ("train/loss", "Train loss", "#1f77b4", "-"),
    ("eval/loss", "Eval loss", "#ff7f0e", "-"),
])

_GRPO_PANES: List[_PaneSpec] = [
    ("Reward", "Reward", [
        ("train/reward", "Train reward", "#2ca02c", "-"),
        ("eval/reward", "Eval reward", "#d62728", "-"),
        ("train/reward_std", "Train reward std", "#9467bd", "--"),
    ]),
    ("Reward-group health", "Fraction", [
        ("train/frac_reward_zero_std", "Train frac zero-std", "#8c564b", "-"),
        ("eval/frac_reward_zero_std", "Eval frac zero-std", "#8c564b", "--"),
    ]),
    ("KL / Entropy", "Value", [
        ("train/kl", "Train KL", "#17becf", "-"),
        ("eval/kl", "Eval KL", "#bcbd22", "--"),
        ("train/entropy", "Train entropy", "#7f7f7f", "-"),
    ]),
    ("Completion length", "Tokens / ratio", [
        ("train/completions/mean_length", "Mean length", "#1f77b4", "-"),
        ("train/completions/clipped_ratio", "Clipped ratio", "#ff7f0e", "-"),
    ]),
]

_DPO_PANES: List[_PaneSpec] = [
    ("DPO implicit rewards", "Reward", [
        ("train/rewards/chosen", "Train chosen", "#2ca02c", "-"),
        ("train/rewards/rejected", "Train rejected", "#d62728", "-"),
        ("eval/rewards/chosen", "Eval chosen", "#2ca02c", "--"),
        ("eval/rewards/rejected", "Eval rejected", "#d62728", "--"),
    ]),
    ("Preference accuracy & margin", "Value", [
        ("train/rewards/accuracies", "Train accuracy", "#9467bd", "-"),
        ("eval/rewards/accuracies", "Eval accuracy", "#9467bd", "--"),
        ("train/rewards/margins", "Train margin", "#8c564b", "-"),
        ("eval/rewards/margins", "Eval margin", "#8c564b", "--"),
    ]),
    ("Log-probs (chosen vs rejected)", "logp", [
        ("train/logps/chosen", "Train chosen", "#17becf", "-"),
        ("train/logps/rejected", "Train rejected", "#bcbd22", "-"),
    ]),
]

_TAIL_PANES: List[_PaneSpec] = [
    ("Gradient norm", "Norm", [
        ("train/grad_norm", "Grad norm", "#e377c2", "-"),
    ]),
    ("Learning rate", "LR", [
        ("train/learning_rate", "Learning rate", "#1f77b4", "-"),
    ]),
]


def _build_panes(tags: set, method: Optional[str]) -> List[_PaneSpec]:
    """Method-aware pane specs. Unknown method offers BOTH sets; empty panes are dropped later."""
    detected = _detect_method(tags, method)
    panes: List[_PaneSpec] = [_LOSS_PANE]
    if detected in ("grpo", "unknown"):
        panes.extend(_GRPO_PANES)
    if detected in ("dpo", "unknown"):
        panes.extend(_DPO_PANES)
    panes.extend(_TAIL_PANES)
    return panes


def plot_iteration_metrics(log_root: str, *, smooth_window: int = 2,
                           method: Optional[str] = None) -> Any:
    """Render the cross-iteration training dashboard for a finished (or running) arm.

    Args:
        log_root: a directory containing event files, recursed -- normally
            ``data/runs/<EXP_NAME>/``, which holds one ``iteration_N/training/tb_logs/`` per
            iteration. ``tb_live/`` is skipped (its steps are already cumulative; view it in the TB
            web UI instead).
        smooth_window: trailing moving-average window, for readability only. The learning-rate pane
            ignores it -- a smoothed LR schedule is a lie.
        method: ``'grpo'`` | ``'pto'`` | ``'dpo'`` to force the pane set; ``None`` auto-detects from
            the tags TRL actually wrote (``rewards/chosen`` => DPO, ``reward_std`` /
            ``frac_reward_zero_std`` => GRPO) and offers both when it cannot tell.

    Returns:
        The matplotlib ``Figure``, or ``None`` when there is nothing to draw.

    Notes:
        **Degrades, never raises.** A missing tag drops its series, a pane with no surviving series
        is dropped, and a run with no event files at all prints a warning and returns ``None`` --
        this is called on partially-trained arms and on arms whose Drive mount is half-synced.

        Dotted vertical lines mark iteration boundaries; the x axis is the re-based cumulative step
        from :func:`parse_tensorboard_logs`, so the curve chains end to end across iterations.
    """
    import matplotlib.pyplot as plt

    df = parse_tensorboard_logs(log_root)
    if df is None or df.empty:
        print(f"  [tb] no scalars under {os.path.abspath(str(log_root))}; nothing to plot")
        return None

    boundaries = compute_iteration_boundaries(df)
    tags = set(df["tag"].unique())
    print(f"  [tb] {len(tags)} scalar tag(s) across {len(boundaries)} iteration(s)")

    resolved: List[Tuple[str, str, List[Tuple[str, str, str, str]]]] = []
    for title, ylabel, specs in _build_panes(tags, method):
        present: List[Tuple[str, str, str, str]] = []
        seen: set = set()
        for want, label, color, ls in specs:
            tag = _resolve_tag(tags, want)
            if tag is None or tag in seen:
                continue
            seen.add(tag)
            present.append((tag, label, color, ls))
        if present:
            resolved.append((title, ylabel, present))

    if not resolved:
        print("  [tb] no known metric families present; run scan_scalar_tags to see what is there")
        return None

    ncols = 2
    nrows = (len(resolved) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4.5 * nrows), squeeze=False)
    fig.suptitle(f"Training dashboard - {os.path.basename(os.path.abspath(str(log_root)))}",
                 fontsize=15)

    for idx, (title, ylabel, present) in enumerate(resolved):
        ax = axes[idx // ncols][idx % ncols]
        plotted = False
        for tag, label, color, ls in present:
            sub = df[df["tag"] == tag].sort_values("global_step")
            xs = [float(v) for v in sub["global_step"].tolist()]
            ys = [float(v) for v in sub["value"].tolist()]
            if not xs:
                continue
            window = 1 if "learning_rate" in tag else max(1, int(smooth_window))
            xs, ys = _smooth(xs, ys, window)
            ax.plot(xs, ys, marker="o", markersize=3, linewidth=1.6,
                    label=label, color=color, linestyle=ls)
            plotted = True
        for b in boundaries[:-1]:                      # the last boundary is the plot's right edge
            ax.axvline(b, color="gray", alpha=0.3, linewidth=0.8, linestyle=":")
        ax.set_title(title)
        ax.set_xlabel("Cumulative step")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        if plotted:
            ax.legend(fontsize=9)
        else:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)

    for j in range(len(resolved), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.tight_layout()
    plt.show()
    print(f"  [tb] rendered {len(resolved)} pane(s)")
    return fig
