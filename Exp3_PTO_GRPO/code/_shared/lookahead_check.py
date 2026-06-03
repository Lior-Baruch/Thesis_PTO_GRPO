"""lookahead_check.py — optional serial-vs-batched look-ahead validation.

NOT on the trainer hot path. Run this from a notebook (after the policy +
tokenizer + OpenAI client are loaded) to confirm that the batched rewrite
:func:`reward.simulate_lookahead_batch` matches the legacy serial
:func:`reward.simulate_lookahead_single` *in distribution* (they are not
bit-identical — sampling RNG differs), and to exercise the OOM sub-batch-halving
path.

Two helpers:
- :func:`make_quick_fixtures` — generate a handful of short conversations with the
  current policy and turn them into ``(transcript, completion, patient_system_prompt)``
  fixtures (transcript ends on a patient turn; completion is the conversation's real
  next therapist turn — exactly the shape the reward fn extends).
- :func:`compare_serial_vs_batched` — run both look-ahead paths on the same fixtures,
  report wall-clock, realized-turn counts, and (optionally) Q1+Q2 oracle reward
  mean/std for each, plus the batched GPU-call count and final sub-batch.

Typical use (notebook cell)::

    from _shared.lookahead_check import make_quick_fixtures, compare_serial_vs_batched
    fx = make_quick_fixtures(
        policy=base_policy, tokenizer=tokenizer, client=client,
        permutations=all_permutations[:8],
        therapist_system_prompt=therapist_system_prompt,
        therapist_init_utterance=therapist_init_utterance,
        cfg=cfg, n_fixtures=48,
    )
    res = await compare_serial_vs_batched(
        **fx, system_prompt_therapist=therapist_system_prompt,
        therapist_model=base_policy, therapist_tokenizer=tokenizer, client=client,
        lookahead_cfg=lookahead_cfg, primitives=primitives,
        oracle_cfg=oracle_cfg, questionnaire_ids=QUESTIONNAIRE_IDS,
        sub_batch_size=LOOKAHEAD_SUB_BATCH_SIZE,
    )
"""

import time
import asyncio
from typing import List, Optional, Sequence, Dict

import numpy as np

from .convs import (
    generate_all_conversations,
    turns_to_messages,
    format_conversation_for_oracle,
)
from .reward import (
    simulate_lookahead_single,
    simulate_lookahead_batch,
    LookaheadConfig,
    OracleConfig,
    OracleAsyncPrimitives,
    _process_single_sample,
)

__all__ = ["make_quick_fixtures", "compare_serial_vs_batched"]


def _count_labels(s: str) -> int:
    return s.count("[PATIENT]:") + s.count("[THERAPIST]:")


def make_quick_fixtures(
    *,
    policy,
    tokenizer,
    client,
    permutations: List[Dict],
    therapist_system_prompt: str,
    therapist_init_utterance: str,
    cfg,
    n_fixtures: int = 48,
    fixtures_per_conv: int = 8,
    num_utterances: int = 31,
) -> Dict[str, list]:
    """Generate short convs with *policy* and derive look-ahead fixtures.

    Each fixture is a ``(transcript, completion, patient_system_prompt)`` triple
    where ``transcript`` ends on a patient turn and ``completion`` is the
    conversation's actual next therapist turn. Patient prompts are non-empty (so
    the degenerate-patient guard in the reward fn is not tripped).

    Returns a dict with keys ``transcripts``, ``completions``,
    ``system_prompts_patient`` (ready to splat into
    :func:`compare_serial_vs_batched`).
    """
    policy.eval()
    policy.config.use_cache = True

    states = generate_all_conversations(
        therapist_model=policy,
        therapist_tokenizer=tokenizer,
        client=client,
        permutations=permutations,
        therapist_system_prompt=therapist_system_prompt,
        therapist_init_utterance=therapist_init_utterance,
        save_dir=None,
        patient_model_id=cfg.patient_model_id,
        max_tokens_per_response=cfg.max_tokens_per_response,
        num_utterances=num_utterances,
        temperature_therapist=cfg.temperature_therapist_gen,
        temperature_patient=cfg.temperature_patient,
        batch_size=cfg.conversation_batch_size,
        therapist_max_input_tokens=cfg.therapist_max_input_tokens,
        patient_api_concurrency=cfg.patient_api_concurrency,
        patient_api_max_retries=cfg.patient_api_max_retries,
        patient_api_backoff_seconds=cfg.patient_api_backoff_seconds,
        batch_cooldown_seconds=1.0,
        max_retries_without_progress=cfg.max_gen_retries_without_progress,
        stop_strings=cfg.stop_strings,
        verbose=False,
        verbose_detailed=False,
    )

    transcripts: List[str] = []
    completions: List[str] = []
    patient_sps: List[str] = []

    for state in states:
        turns = state.turns
        if not turns and state.conversation:
            turns = [
                {"role": "therapist" if j % 2 == 0 else "patient", "content": utt}
                for j, utt in enumerate(state.conversation)
            ]
        if not turns:
            continue
        patient_sp = permutations[state.permutation_index]["patient_system_prompt"]

        taken = 0
        # Slice after a patient turn whose next turn is therapist (= the completion).
        for i, turn in enumerate(turns[:-1]):
            if turn["role"] != "patient":
                continue
            nxt = turns[i + 1]
            if nxt["role"] != "therapist" or not (nxt["content"] or "").strip():
                continue
            if (i + 1) < cfg.min_conv_length:
                continue
            messages = turns_to_messages(turns[: i + 1], therapist_system_prompt)
            transcripts.append(format_conversation_for_oracle(messages))
            completions.append(nxt["content"])
            patient_sps.append(patient_sp)
            taken += 1
            if taken >= fixtures_per_conv or len(transcripts) >= n_fixtures:
                break
        if len(transcripts) >= n_fixtures:
            break

    print(
        f"  Built {len(transcripts)} look-ahead fixtures from "
        f"{len(states)} generated conversations."
    )
    return {
        "transcripts": transcripts,
        "completions": completions,
        "system_prompts_patient": patient_sps,
    }


async def _score(extended: List[str], client, oracle_cfg, primitives, questionnaire_ids):
    """Oracle-score a list of extended transcripts (Q1+Q2 mean per sample)."""
    stats = {"success": 0, "fail": 0}
    tasks = [
        _process_single_sample(
            client, oracle_cfg, primitives, questionnaire_ids, stats,
            idx, "", "", full_conversation_override=ext,
        )
        for idx, ext in enumerate(extended)
    ]
    return await asyncio.gather(*tasks)


async def compare_serial_vs_batched(
    *,
    transcripts: List[str],
    completions: List[str],
    system_prompts_patient: List[str],
    system_prompt_therapist: str,
    therapist_model,
    therapist_tokenizer,
    client,
    lookahead_cfg: LookaheadConfig,
    primitives: OracleAsyncPrimitives,
    oracle_cfg: Optional[OracleConfig] = None,
    questionnaire_ids: Optional[Sequence[int]] = None,
    sub_batch_size: Optional[int] = None,
) -> dict:
    """Run serial vs batched look-ahead on the same fixtures and compare.

    Both paths self-toggle ``eval()`` / ``use_cache`` internally, so the model
    may be in either mode on entry. Compares **distributions** (realized turns,
    optional Q1+Q2 oracle mean/std), not strings — therapist sampling RNG differs
    between the two paths.

    If ``oracle_cfg`` and ``questionnaire_ids`` are given, the equivalence target
    is ``|mean_batched - mean_serial|`` within the oracle reproducibility noise
    (~0.07-0.10 per Partial_Conv_Oracle_EDA). Returns a dict of all measurements.
    """
    if lookahead_cfg.k <= 0:
        raise ValueError("compare_serial_vs_batched requires lookahead_cfg.k > 0.")

    n = len(transcripts)
    seeds = [f"{t}\n\n[THERAPIST]: {c}" for t, c in zip(transcripts, completions)]

    # ── Serial (legacy, batch-of-1) ──
    t0 = time.time()
    serial_ext = await asyncio.gather(*[
        simulate_lookahead_single(
            transcript=t, completion=c,
            system_prompt_therapist=system_prompt_therapist,
            system_prompt_patient=sp,
            therapist_model=therapist_model,
            therapist_tokenizer=therapist_tokenizer,
            client=client,
            patient_model_id=lookahead_cfg.patient_model_id,
            lookahead_k=lookahead_cfg.k,
            temperature_therapist=lookahead_cfg.temperature_therapist,
            temperature_patient=lookahead_cfg.temperature_patient,
            max_tokens=lookahead_cfg.max_tokens,
            max_input_tokens=lookahead_cfg.max_input_tokens,
            stop_strings=lookahead_cfg.stop_strings,
            patient_sem=primitives.lookahead_patient_sem(),
            gpu_lock=primitives.gpu_lock(),
            patient_max_retries=lookahead_cfg.patient_api_max_retries,
            patient_backoff_seconds=lookahead_cfg.patient_api_backoff_seconds,
        )
        for t, c, sp in zip(transcripts, completions, system_prompts_patient)
    ])
    serial_t = time.time() - t0

    # ── Batched (lock-step) ──
    tele: dict = {}
    t1 = time.time()
    batched_ext = await simulate_lookahead_batch(
        transcripts=transcripts,
        completions=completions,
        system_prompt_therapist=system_prompt_therapist,
        system_prompts_patient=system_prompts_patient,
        therapist_model=therapist_model,
        therapist_tokenizer=therapist_tokenizer,
        client=client,
        patient_model_id=lookahead_cfg.patient_model_id,
        lookahead_k=lookahead_cfg.k,
        temperature_therapist=lookahead_cfg.temperature_therapist,
        temperature_patient=lookahead_cfg.temperature_patient,
        max_tokens=lookahead_cfg.max_tokens,
        max_input_tokens=lookahead_cfg.max_input_tokens,
        stop_strings=lookahead_cfg.stop_strings,
        patient_sem=primitives.lookahead_patient_sem(),
        gpu_lock=primitives.gpu_lock(),
        patient_max_retries=lookahead_cfg.patient_api_max_retries,
        patient_backoff_seconds=lookahead_cfg.patient_api_backoff_seconds,
        sub_batch_size=sub_batch_size,
        telemetry=tele,
    )
    batched_t = time.time() - t1

    serial_turns = [_count_labels(e) - _count_labels(s) for e, s in zip(serial_ext, seeds)]
    batched_turns = [_count_labels(e) - _count_labels(s) for e, s in zip(batched_ext, seeds)]

    out = {
        "n_fixtures": n,
        "k": lookahead_cfg.k,
        "serial_seconds": serial_t,
        "batched_seconds": batched_t,
        "speedup": (serial_t / batched_t) if batched_t > 0 else float("nan"),
        "serial_avg_turns": float(np.mean(serial_turns)) if serial_turns else 0.0,
        "batched_avg_turns": float(np.mean(batched_turns)) if batched_turns else 0.0,
        "batched_gpu_calls": tele.get("gpu_calls"),
        "batched_final_sub_batch": tele.get("sub_batch"),
    }

    print("\n── Look-ahead serial vs batched ──")
    print(f"  fixtures={n}  K={lookahead_cfg.k}  sub_batch_size={sub_batch_size}")
    print(f"  serial : {serial_t:6.1f}s   avg realized turns {out['serial_avg_turns']:.2f}")
    print(f"  batched: {batched_t:6.1f}s   avg realized turns {out['batched_avg_turns']:.2f}"
          f"   ({tele.get('gpu_calls')} GPU calls, final sub_batch={tele.get('sub_batch')})")
    print(f"  speedup: {out['speedup']:.1f}x")

    if oracle_cfg is not None and questionnaire_ids is not None:
        serial_r = [r for r in await _score(serial_ext, client, oracle_cfg, primitives, questionnaire_ids)
                    if r is not None]
        batched_r = [r for r in await _score(batched_ext, client, oracle_cfg, primitives, questionnaire_ids)
                     if r is not None]
        s_mean = float(np.mean(serial_r)) if serial_r else float("nan")
        b_mean = float(np.mean(batched_r)) if batched_r else float("nan")
        out.update({
            "serial_reward_mean": s_mean,
            "serial_reward_std": float(np.std(serial_r)) if serial_r else float("nan"),
            "batched_reward_mean": b_mean,
            "batched_reward_std": float(np.std(batched_r)) if batched_r else float("nan"),
            "abs_mean_gap": abs(b_mean - s_mean),
        })
        print(f"  serial  Q1Q2 reward: mean {out['serial_reward_mean']:.3f} "
              f"std {out['serial_reward_std']:.3f}  (n={len(serial_r)})")
        print(f"  batched Q1Q2 reward: mean {out['batched_reward_mean']:.3f} "
              f"std {out['batched_reward_std']:.3f}  (n={len(batched_r)})")
        print(f"  |Δmean| = {out['abs_mean_gap']:.3f}  "
              f"(equivalent if within oracle noise ~0.07-0.10)")

    return out
