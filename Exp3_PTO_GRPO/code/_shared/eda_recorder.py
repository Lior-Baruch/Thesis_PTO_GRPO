"""
eda_recorder.py — per-generation EDA capture, shared by GRPO_Exp3 and PTO_Exp3.

Both trainers throw away almost everything the policy generates: PTO keeps only
the final (chosen, rejected) pair per branch, GRPO keeps nothing per-prompt. For
the thesis EDA (reward-faithfulness / partial-conv analysis) we want **every**
candidate the model produced each iteration, with its oracle score, the
branch/group it belongs to, the K-turn look-ahead transcript the oracle actually
scored, per-questionnaire sub-scores, and oracle/look-ahead flags.

``EDARecorder`` is a tiny in-memory buffer with an atomic, once-per-iteration
flush. The hot path only calls :meth:`append` (a cheap list append — safe to call
from inside the async reward fn without blocking the event loop or hitting the
Colab Drive-FUSE mount); the single :meth:`flush` writes
``iteration_N/eda/generations.jsonl`` at the end of the iteration.

One unified **branch-centric** schema serves all three paths (GRPO, PTO greedy,
PTO independent): one record per branch, the prefix stored **once** (oracle-format
transcript), the candidates **nested**, and the look-ahead stored as the **tail**
only (the K simulated turns — prefix+completion sliced off). The full base
conversations live in the already-saved ``model_iter_*`` eval convs::

    {
      "method": "PTO_Exp3" | "GRPO_Exp3", "iteration": 3,
      "phase": "tree" | "independent" | "train" | "eval",
      "conversation_id": 17,            # permutation_index
      "branch_id": 24,                  # PTO: branch_depth / branch_turn_index; GRPO: running group counter
      "epoch": 1.0 | None,              # GRPO only (trainer_state.epoch); None for PTO
      "prefix": "[PATIENT]: ...\n\n[THERAPIST]: ...",   # oracle transcript of conv-so-far, ONCE
      "group_mean": 2.41 | None, "group_std": 0.52 | None,   # GRPO only
      "chosen_idx": 7 | None,           # argmax score = the followed "first therapist answer"
      "candidates": [
        {
          "idx": 0, "completion": "...",
          "score": 2.1 | None,                          # Q1+Q2 mean (None on oracle fail)
          "sub_scores": {"1": 2.0, "2": 2.2} | None,    # per-questionnaire mean_score
          "role": "chosen" | "rejected" | "neither" | None,   # PTO only
          "oracle": {"success": True, "retries": 0},
          "lookahead": {"k": 5, "realized_turns": 4, "ended_early": True, "tail": "..."|None},
        }, ...
      ],
    }

Reconstruct the oracle-scored text of a candidate as
``prefix + "\\n\\n[THERAPIST]: " + completion + (tail or "")``.
"""

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np


def _to_jsonable(o):
    """json.dump default: coerce numpy scalars so records are always serializable."""
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


class EDARecorder:
    """In-memory per-iteration buffer for candidate-level generation records.

    Args:
        out_path: where :meth:`flush` writes the JSONL (one record per line).
        enabled: when False every method is a no-op (zero overhead, nothing written).
        save_transcripts: when False the per-candidate ``lookahead.tail`` (the K
            simulated turns) is dropped at append time (flags + scores still kept).
            The size lever for look-ahead-heavy runs.
    """

    def __init__(self, out_path: str, *, enabled: bool = True, save_transcripts: bool = True):
        self.out_path = out_path
        self.enabled = enabled
        self.save_transcripts = save_transcripts
        self.records: List[dict] = []

    def append(self, record: dict) -> None:
        """Buffer one **branch** record (prefix once + nested ``candidates``).

        Cheap in-memory append. No-op when disabled. When ``save_transcripts=False``
        the per-candidate look-ahead ``tail`` is dropped (flags + scores kept).
        """
        if not self.enabled:
            return
        if not self.save_transcripts:
            for cand in record.get("candidates", []):
                la = cand.get("lookahead")
                if isinstance(la, dict) and la.get("tail") is not None:
                    la["tail"] = None
        self.records.append(record)

    def _write_jsonl(self, path: str) -> str:
        """Atomically write the current buffer to *path* (tmp + os.replace).

        Shared by :meth:`flush` (end-of-iteration → ``out_path``) and
        :meth:`snapshot_to` (per-checkpoint crash-recovery copy). Writes even an
        empty file so its presence is a reliable "this ran" signal.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            for rec in self.records:
                f.write(json.dumps(rec, ensure_ascii=False, default=_to_jsonable))
                f.write("\n")
        os.replace(tmp, path)
        return path

    def flush(self) -> Optional[str]:
        """Atomically write the buffer to ``out_path`` (tmp + os.replace).

        One write per iteration keeps the Colab Drive-FUSE mount happy. Returns the
        path written, or None when disabled.
        """
        if not self.enabled:
            return None
        return self._write_jsonl(self.out_path)

    def snapshot_to(self, path: str) -> Optional[str]:
        """Atomically write the current buffer to *path* (crash-recovery snapshot).

        Called from the checkpoint ``on_save`` callback to drop the in-memory
        records alongside each HF checkpoint, so a mid-iteration crash + resume can
        reload the pre-crash records (see :meth:`load_from`) instead of losing them.
        No-op (returns None) when disabled. Bound to the checkpoint dir so it stays
        aligned with whichever checkpoint resume actually walks back to.
        """
        if not self.enabled:
            return None
        return self._write_jsonl(path)

    def load_from(self, path: str) -> int:
        """Replace the buffer with records read from a JSONL snapshot at *path*.

        Inverse of :meth:`snapshot_to`, called on resume before training restarts so
        the final :meth:`flush` writes pre-crash + post-resume rows. Guarded no-op
        (returns 0) when disabled or the file is missing/unreadable — so existing
        checkpoints without a snapshot behave exactly as before. Returns the number
        of records loaded.
        """
        if not self.enabled or not path or not os.path.exists(path):
            return 0
        try:
            with open(path, "r", encoding="utf-8") as f:
                recs = [json.loads(line) for line in f if line.strip()]
        except (OSError, ValueError):
            return 0
        self.records = recs
        return len(recs)

    # ── Aggregates (for the live TensorBoard / W&B per-iteration scalars) ──

    def aggregate(self) -> Tuple[Dict[str, float], List[float]]:
        """Summarize the buffer into ``(scalars, scores)`` for run-level logging.

        Iterates the **branch** records → their nested candidates. ``scalars`` are
        per-iteration metrics keyed for TB/W&B (shared + method-specific); ``scores``
        is the flat list of non-None candidate rewards for a histogram. A trainer
        just forwards the result to ``RunTBLogger`` without re-deriving anything.
        """
        branches = self.records
        cands = [c for b in branches for c in b.get("candidates", [])]
        n = len(cands)
        scores = [float(c["score"]) for c in cands if c.get("score") is not None]
        scalars: Dict[str, float] = {
            "eda/n_branches": float(len(branches)),
            "eda/n_candidates": float(n),
        }
        if scores:
            scalars["eda/mean_candidate_reward"] = float(np.mean(scores))
            scalars["eda/reward_std"] = float(np.std(scores))
        if n:
            scalars["eda/oracle_success_rate"] = float(
                np.mean([1.0 if (c.get("oracle") or {}).get("success") else 0.0 for c in cands])
            )

        la = [
            c["lookahead"] for c in cands
            if isinstance(c.get("lookahead"), dict) and (c["lookahead"].get("k") or 0) > 0
        ]
        if la:
            scalars["eda/lookahead_realized_turns_mean"] = float(
                np.mean([d.get("realized_turns", 0) for d in la])
            )
            scalars["eda/lookahead_ended_early_frac"] = float(
                np.mean([1.0 if d.get("ended_early") else 0.0 for d in la])
            )

        # ── PTO-specific: one branch row each; a pair was emitted iff the branch
        #    contains a "rejected" candidate. ──
        pto_branches = [b for b in branches if b.get("method") == "PTO_Exp3"]
        if pto_branches:
            bp = len(pto_branches)
            pp = sum(
                1 for b in pto_branches
                if any(c.get("role") == "rejected" for c in b.get("candidates", []))
            )
            scalars["pto/branch_points"] = float(bp)
            scalars["pto/pref_pair_count"] = float(pp)
            if bp:
                scalars["pto/tau_filter_rate"] = float(1.0 - pp / bp)

        # ── GRPO-specific: group std/mean live on the branch record. ──
        grpo_stds = [float(b["group_std"]) for b in branches if b.get("group_std") is not None]
        if grpo_stds:
            scalars["grpo/group_reward_std_mean"] = float(np.mean(grpo_stds))
            scalars["grpo/frac_zero_std"] = float(np.mean([1.0 if v == 0.0 else 0.0 for v in grpo_stds]))
            scalars["grpo/num_groups"] = float(len(grpo_stds))

        return scalars, scores

    def sample_for_display(self, n: int) -> List[dict]:
        """Pick up to ``n`` candidates spread across the score range (best/median/worst).

        Flattens the nested candidates and attaches each one's branch ``prefix`` +
        group context, shaped for ``RunTBLogger.log_sample_completions`` (which reads
        ``prompt``/``completion``/``score``/``pto``/``grpo``/``lookahead``).
        """
        flat: List[dict] = []
        for b in self.records:
            for c in b.get("candidates", []):
                flat.append({
                    "prompt": b.get("prefix"),
                    "completion": c.get("completion"),
                    "score": c.get("score"),
                    "sub_scores": c.get("sub_scores"),
                    "lookahead": c.get("lookahead"),
                    "pto": ({"role": c.get("role")} if c.get("role") is not None else None),
                    "grpo": (
                        {"group_mean": b.get("group_mean"), "group_std": b.get("group_std")}
                        if b.get("group_mean") is not None else None
                    ),
                })
        scored = [r for r in flat if r.get("score") is not None]
        if not scored or n <= 0:
            return flat[:n] if n > 0 else []
        scored.sort(key=lambda r: r["score"])
        if n >= len(scored):
            return scored
        # Evenly sample indices across the sorted range so we span worst→best.
        idxs = sorted({int(round(i * (len(scored) - 1) / (n - 1))) for i in range(n)})
        return [scored[i] for i in idxs]
