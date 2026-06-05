"""
tb_plots.py — Logging lifecycle + TensorBoard parsing & plots.

Three concerns share this module:

1. **Per-iteration logging lifecycle** — ``init_iteration_logging`` /
   ``finish_iteration_logging`` start/stop the W&B side of each iteration,
   while TB is auto-wired by HuggingFace via the env var below. Trainer
   callbacks ``CheckpointMetadataCallback`` / ``CumulativeStepCallback`` are
   in this module too, shared between GRPO and PTO trainers.
2. **TensorBoard setup** — ``setup_tensorboard_logging`` /
   ``patch_trainer_tensorboard_callback`` ensure the Trainer writes to an
   explicit, Windows-safe log directory.
3. **Post-run analysis** — ``parse_tensorboard_logs`` / ``plot_iteration_metrics``
   are the dashboard tools the notebook uses to inspect a finished run.

Per-iteration step offsets: each iteration's trainer (``GRPOTrainer`` or
``DPOTrainer``) writes its TB event file starting at step 0. To plot
continuous curves across iterations, we extract the iteration number from each
event file's path (``iteration_N/training/tb_logs/...``) and shift its steps
by the sum of completed steps in earlier iterations. Without this, iter1 step
50 and iter2 step 50 collide on dedup and the curve loses data.
"""

import json
import math
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb
from tensorboard.backend.event_processing import event_accumulator as ea
from transformers import TrainerCallback


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                          TRAINER CALLBACKS                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


class CheckpointMetadataCallback(TrainerCallback):
    """Write ``experiment_metadata.json`` alongside each saved epoch checkpoint."""

    def __init__(self, iteration: int, metadata: dict):
        self.iteration = iteration
        self.metadata = metadata

    def on_save(self, args, state, control, **kwargs):
        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if not os.path.isdir(checkpoint_dir):
            return
        payload = {
            **self.metadata,
            "global_step": state.global_step,
            "epoch": state.epoch,
            "timestamp": pd.Timestamp.now().isoformat(),
        }
        with open(os.path.join(checkpoint_dir, "experiment_metadata.json"), "w") as f:
            json.dump(payload, f, indent=2)


class CumulativeStepCallback(TrainerCallback):
    """Inject ``cumulative_global_step`` into logs for cross-iteration continuity.

    Each new trainer resets ``global_step`` to 0; the offset keeps the W&B
    x-axis continuous across iterations. No-op for TensorBoard / none (TB
    cross-iter continuity is reconstructed at plot time via the event-file
    path-based offsets in :func:`parse_tensorboard_logs`).
    """

    def __init__(self, step_offset: int, report_to):
        self.step_offset = step_offset
        self.active = "wandb" in report_to

    def on_log(self, args, state, control, logs=None, **kwargs):
        if self.active and logs is not None:
            logs["cumulative_global_step"] = state.global_step + self.step_offset


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                  UNIFIED LOGGING LIFECYCLE                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


def init_iteration_logging(report_to, iteration: int, cumulative_step_offset: int, wandb_ctx):
    """Initialize the logging backend for this iteration's training phase.

    Called just before ``trainer.train()``.

    - wandb: open (or resume) a single run once, then reuse it.
    - tensorboard / none: no-op (HF Trainer auto-creates the SummaryWriter).
    """
    if "wandb" in report_to and wandb_ctx is not None:
        if wandb.run is None:
            wandb.init(
                project=wandb_ctx["project"],
                name=wandb_ctx["run_id"],
                id=wandb_ctx["run_id"],
                resume="allow",
                config=wandb_ctx["config"],
            )
            wandb.define_metric("cumulative_global_step")
            wandb.define_metric("*", step_metric="cumulative_global_step")
            print(f"  ✓ W&B initialized (run_id={wandb_ctx['run_id']})")
        else:
            print(f"  ✓ W&B run already active: {wandb.run.id}")

    if "tensorboard" in report_to:
        print("  ✓ TensorBoard logging to logging_dir set in trainer config")


def finish_iteration_logging(report_to, iteration: int, iter_level_metrics: dict):
    """Log iteration-level summary metrics.

    Called after adapter save. W&B is intentionally kept open so the whole
    experiment stays on a single run with a continuous step axis.
    """
    if "wandb" in report_to and wandb.run is not None:
        wandb.log(iter_level_metrics)
        print(f"  ✓ W&B iteration metrics logged for iteration {iteration}")
    if "tensorboard" in report_to:
        # Iteration-level metrics are already in the checkpoint metadata JSON.
        # TensorBoard scalars come from the Trainer's built-in logging.
        print(f"  ✓ TensorBoard logs written for iteration {iteration}")


def setup_tensorboard_logging(report_to, tensorboard_log_dir: Optional[str]) -> None:
    """Configure TensorBoard log directory via environment variable.

    Forces TensorBoard to use an explicit, Windows-safe log directory.
    transformers' ``TensorBoardCallback`` reads ``TENSORBOARD_LOGGING_DIR``
    at init.
    """
    if "tensorboard" not in report_to:
        os.environ.pop("TENSORBOARD_LOGGING_DIR", None)
        return
    if tensorboard_log_dir is None:
        raise ValueError("tensorboard_log_dir is required when reporting to tensorboard")
    os.makedirs(tensorboard_log_dir, exist_ok=True)
    os.environ["TENSORBOARD_LOGGING_DIR"] = tensorboard_log_dir
    print(f"  ✓ TensorBoard log dir: {tensorboard_log_dir}")


def patch_trainer_tensorboard_callback(trainer, tensorboard_log_dir: str) -> None:
    """Patch the TensorBoardCallback on a trainer to use our explicit log dir.

    Ensures the callback writes to the correct directory even when default
    path resolution would produce something unusable (e.g. WinError 123 on
    Windows from default_logdir()).
    """
    for cb in trainer.callback_handler.callbacks:
        if cb.__class__.__name__ == "TensorBoardCallback":
            cb.logging_dir = tensorboard_log_dir
            if getattr(cb, "tb_writer", None) is not None:
                cb.tb_writer.close()
                cb.tb_writer = None
            print(f"  ✓ TensorBoard callback patched → {tensorboard_log_dir}")
            return
    print("  ⚠ TensorBoardCallback not found on trainer")


_ITER_DIR_RE = re.compile(r"iteration_(\d+)")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                          EVENT-FILE PARSING                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


def find_event_files(log_root) -> List[Path]:
    """Return every ``events.out.tfevents.*`` under ``log_root`` (sorted)."""
    return sorted(Path(log_root).rglob("events.out.tfevents.*"))


def _extract_iter_from_path(path: Path) -> int:
    """Extract iteration number from an event-file path under ``iteration_N/...``.

    Returns 0 if no ``iteration_N`` segment is found (treated as a pre-iter
    or standalone log).
    """
    for part in path.parts:
        m = _ITER_DIR_RE.fullmatch(part)
        if m:
            return int(m.group(1))
    return 0


def parse_tensorboard_logs(log_dir) -> Dict[str, List[Tuple[int, float]]]:
    """Return ``{tag: [(cumulative_step, value), ...]}`` deduplicated by step.

    Each event file's local steps are offset by the total steps completed in
    earlier iterations so cross-iteration curves chain end-to-end. The offset
    for iteration N is the sum of max-step values from iterations 1..N-1
    (one event file per iteration's training run).
    """
    event_files = find_event_files(log_dir)

    # ── Pass 1: read raw scalars per file, record per-iteration max step ──
    raw: List[Tuple[int, Dict[str, List[Tuple[int, float]]], int]] = []
    for event_file in event_files:
        it = _extract_iter_from_path(event_file)
        try:
            acc = ea.EventAccumulator(str(event_file), size_guidance={ea.SCALARS: 0})
            acc.Reload()
            per_tag: Dict[str, List[Tuple[int, float]]] = {}
            max_step = 0
            for tag in acc.Tags().get("scalars", []):
                events = acc.Scalars(tag)
                per_tag[tag] = [(e.step, e.value) for e in events]
                if events:
                    max_step = max(max_step, max(e.step for e in events))
            raw.append((it, per_tag, max_step))
        except Exception as e:
            print(f"  Warning: Could not read {event_file}: {e}")
            continue

    # ── Build cumulative step offset per iteration ──
    per_iter_max: Dict[int, int] = {}
    for it, _, ms in raw:
        per_iter_max[it] = max(per_iter_max.get(it, 0), ms)
    offsets: Dict[int, int] = {}
    cumulative = 0
    for it in sorted(per_iter_max.keys()):
        offsets[it] = cumulative
        cumulative += per_iter_max[it]

    # ── Pass 2: apply offsets and merge tags ──
    metrics: Dict[str, List[Tuple[int, float]]] = {}
    for it, per_tag, _ in raw:
        offset = offsets.get(it, 0)
        for tag, series in per_tag.items():
            metrics.setdefault(tag, []).extend((s + offset, v) for s, v in series)

    # Dedup by cumulative step (collisions impossible after offsetting unless
    # the same iteration's event file is duplicated, which TB doesn't do).
    for tag, series in metrics.items():
        dedup = {step: value for step, value in series}
        metrics[tag] = sorted(dedup.items(), key=lambda x: x[0])
    return metrics


def compute_iteration_boundaries(log_dir) -> List[Tuple[int, int]]:
    """Return ``[(iter_number, cumulative_step_at_end), ...]`` per iteration.

    Used to draw vertical separators at iteration boundaries on cross-iter plots.
    """
    event_files = find_event_files(log_dir)
    per_iter_max: Dict[int, int] = {}
    for event_file in event_files:
        it = _extract_iter_from_path(event_file)
        try:
            acc = ea.EventAccumulator(str(event_file), size_guidance={ea.SCALARS: 0})
            acc.Reload()
            ms = 0
            for tag in acc.Tags().get("scalars", []):
                events = acc.Scalars(tag)
                if events:
                    ms = max(ms, max(e.step for e in events))
            per_iter_max[it] = max(per_iter_max.get(it, 0), ms)
        except Exception:
            continue

    boundaries: List[Tuple[int, int]] = []
    cumulative = 0
    for it in sorted(per_iter_max.keys()):
        cumulative += per_iter_max[it]
        boundaries.append((it, cumulative))
    return boundaries


def scan_scalar_tags(log_root, head: int = 8) -> None:
    """Print a quick scan: event files found + first ``head`` scalar tags per file."""
    log_root = Path(log_root)
    event_files = find_event_files(log_root)
    print(f"Log root: {log_root}")
    print(f"Event files found: {len(event_files)}")
    if not event_files:
        print("⚠ No event files found under log root.")
        return

    all_scalar_tags = set()
    for event_file in event_files:
        try:
            acc = ea.EventAccumulator(str(event_file), size_guidance={ea.SCALARS: 0})
            acc.Reload()
            tags = acc.Tags().get("scalars", [])
            all_scalar_tags.update(tags)
            print(f"\n{event_file}")
            print(f"  Scalars: {len(tags)} tag(s)")
            for tag in tags[:head]:
                print(f"    - {tag}")
            if len(tags) > head:
                print(f"    ... and {len(tags) - head} more")
        except Exception as e:
            print(f"⚠ Could not inspect {event_file}: {e}")

    print("\n" + "=" * 70)
    print(f"Total unique scalar tags: {len(all_scalar_tags)}")


def summarize_available_tags(metrics: Dict[str, list]) -> Dict[str, List[str]]:
    """Bucket the parsed scalar tags by family and print, flagging ``other`` loudly.

    ``metrics`` is the dict from :func:`parse_tensorboard_logs`. The point is to make
    sure no useful DPO/GRPO tag is silently dropped by the dashboard — anything that
    lands in ``other`` is printed as a hint to add it to a pane.
    """
    buckets: Dict[str, List[str]] = {
        "loss": [], "reward": [], "rewards/*": [], "logps/*": [], "logits/*": [],
        "kl/entropy": [], "clip_ratio*": [], "completions/*": [], "lr": [],
        "eda/pto/grpo": [], "other": [],
    }
    for tag in sorted(metrics.keys()):
        t = tag.lower()
        if "loss" in t:
            buckets["loss"].append(tag)
        elif "rewards/" in t:
            buckets["rewards/*"].append(tag)
        elif t.startswith(("eda/", "pto/", "grpo/")):
            buckets["eda/pto/grpo"].append(tag)
        elif "reward" in t:
            buckets["reward"].append(tag)
        elif "logps" in t:
            buckets["logps/*"].append(tag)
        elif "logits" in t:
            buckets["logits/*"].append(tag)
        elif "kl" in t or "entropy" in t:
            buckets["kl/entropy"].append(tag)
        elif "clip_ratio" in t:
            buckets["clip_ratio*"].append(tag)
        elif "completion" in t:
            buckets["completions/*"].append(tag)
        elif "learning_rate" in t or t.endswith("/lr") or t == "lr":
            buckets["lr"].append(tag)
        else:
            buckets["other"].append(tag)

    print("Available TB scalar tags by family:")
    for fam, tags in buckets.items():
        if tags:
            print(f"  {fam}: {len(tags)}")
            for tg in tags:
                print(f"     - {tg}")
    if buckets["other"]:
        print(
            f"\n  ⚠ {len(buckets['other'])} tag(s) not in a known family — consider "
            f"adding to the dashboard: {buckets['other']}"
        )
    return buckets


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                          PLOT UTILITIES                                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


def resolve_tag(
    metrics: Dict[str, list],
    preferred: str,
    fallback_contains: Optional[str] = None,
) -> Optional[str]:
    """Return ``preferred`` if present, else first tag containing ``fallback_contains``."""
    if preferred in metrics:
        return preferred
    if fallback_contains:
        candidates = [t for t in metrics if fallback_contains in t]
        if candidates:
            return sorted(candidates)[0]
    return None


def smooth_xy(series: List[Tuple[int, float]], window: int = 1):
    """Moving-average smoothing for visualization only."""
    if not series:
        return [], []
    xs = [s for s, _ in series]
    ys = [v for _, v in series]
    if window <= 1 or len(ys) < window:
        return xs, ys

    smoothed: List[float] = []
    run = 0.0
    for i, y in enumerate(ys):
        run += y
        if i >= window:
            run -= ys[i - window]
        denom = min(i + 1, window)
        smoothed.append(run / denom)
    return xs, smoothed


def plot_if_present(
    ax,
    metrics: Dict[str, list],
    tag: Optional[str],
    label: str,
    color: Optional[str] = None,
    smooth_window: int = 1,
    linestyle: str = "-",
) -> bool:
    """Plot ``metrics[tag]`` if it exists. Returns ``True`` if plotted."""
    if tag is None or tag not in metrics:
        return False
    xs, ys = smooth_xy(metrics[tag], window=smooth_window)
    ax.plot(xs, ys, marker="o", markersize=3, linewidth=1.8, label=label, color=color, linestyle=linestyle)
    return True


def _annotate_empty(ax, message: str) -> None:
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)


def _draw_iter_boundaries(ax, boundaries: List[Tuple[int, int]]) -> None:
    """Draw a faint vertical line at each iteration's end step."""
    for it, step in boundaries[:-1]:  # skip the last — it's the plot's right edge
        ax.axvline(step, color="gray", alpha=0.3, linewidth=0.8, linestyle=":")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                          TOP-LEVEL DASHBOARD                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


def _build_panes(metrics: Dict[str, list]) -> List[Tuple[str, str, list]]:
    """Method-aware pane specs: ``(title, ylabel, [(tag, label, color, linestyle), ...])``.

    Detects DPO vs GRPO from the tags TRL actually wrote so the dashboard surfaces
    the method-specific metrics the old fixed 2×2 ignored (DPO: rewards/accuracies,
    margins, chosen/rejected, logps; GRPO: reward_std, frac_reward_zero_std,
    completion length). Empty panes are dropped by the caller.
    """
    has_dpo = any("rewards/chosen" in t for t in metrics)
    has_grpo = any(("reward_std" in t) or ("frac_reward_zero_std" in t) for t in metrics)

    panes: List[Tuple[str, str, list]] = []
    panes.append(("Loss", "Loss", [
        ("train/loss", "Train loss", "#1f77b4", "-"),
        ("eval/loss", "Eval loss", "#ff7f0e", "-"),
    ]))

    if has_dpo:
        panes.append(("DPO implicit rewards", "Reward", [
            ("train/rewards/chosen", "Train chosen", "#2ca02c", "-"),
            ("train/rewards/rejected", "Train rejected", "#d62728", "-"),
            ("eval/rewards/chosen", "Eval chosen", "#2ca02c", "--"),
            ("eval/rewards/rejected", "Eval rejected", "#d62728", "--"),
        ]))
        panes.append(("Reward accuracy & margin", "Value", [
            ("train/rewards/accuracies", "Train accuracy", "#9467bd", "-"),
            ("eval/rewards/accuracies", "Eval accuracy", "#9467bd", "--"),
            ("train/rewards/margins", "Train margin", "#8c564b", "-"),
            ("eval/rewards/margins", "Eval margin", "#8c564b", "--"),
        ]))
        panes.append(("Log-probs (chosen vs rejected)", "logp", [
            ("train/logps/chosen", "Train chosen", "#17becf", "-"),
            ("train/logps/rejected", "Train rejected", "#bcbd22", "-"),
        ]))
    elif has_grpo:
        panes.append(("Reward", "Reward", [
            ("train/reward", "Train reward", "#2ca02c", "-"),
            ("eval/reward", "Eval reward", "#d62728", "-"),
            ("train/reward_std", "Train reward std", "#9467bd", "--"),
        ]))
        panes.append(("Reward-group health", "Fraction", [
            ("train/frac_reward_zero_std", "Train frac zero-std", "#8c564b", "-"),
            ("eval/frac_reward_zero_std", "Eval frac zero-std", "#8c564b", "--"),
        ]))
        panes.append(("Completion length", "Tokens / ratio", [
            ("train/completions/mean_length", "Mean length", "#1f77b4", "-"),
            ("train/completions/clipped_ratio", "Clipped ratio", "#ff7f0e", "-"),
        ]))
    else:
        panes.append(("Reward", "Reward", [
            ("train/reward", "Train reward", "#2ca02c", "-"),
            ("eval/reward", "Eval reward", "#d62728", "-"),
        ]))

    # GRPO logs KL + entropy; DPO doesn't → this pane is auto-dropped if empty.
    panes.append(("KL / Entropy", "Value", [
        ("train/kl", "Train KL", "#17becf", "-"),
        ("eval/kl", "Eval KL", "#bcbd22", "--"),
        ("train/entropy", "Train entropy", "#7f7f7f", "-"),
        ("eval/entropy", "Eval entropy", "#e377c2", "--"),
    ]))
    panes.append(("Learning rate", "LR", [
        ("train/learning_rate", "Learning rate", "#1f77b4", "-"),
    ]))
    return panes


def plot_iteration_metrics(log_root, smooth_window: int = 2):
    """Render a method-aware cross-iteration training dashboard.

    Auto-detects DPO (PTO) vs GRPO from the event-file tags and renders the
    metrics each trainer actually logs (so DPO's rewards/accuracies + margins and
    GRPO's reward_std + frac_reward_zero_std are no longer hidden). Per-iteration
    step offsets chain the curves end-to-end; dotted vlines mark iteration
    boundaries. Backward-compatible: falls back to a generic reward pane.

    Args:
        log_root: directory containing ``events.out.tfevents.*`` files (recursed).
                  Point this at the per-iteration ``runs/.../`` tree; for the
                  continuous live curves use the TB web UI on ``tb_live/`` instead.
        smooth_window: moving-average window for visual smoothing only.
    """
    print("Parsing TensorBoard event files...")
    metrics = parse_tensorboard_logs(log_root)
    boundaries = compute_iteration_boundaries(log_root)

    if not metrics:
        print("\n⚠ No metrics found in event files. Check if training has completed.")
        return None

    print(f"Found {len(metrics)} scalar tags across {len(boundaries)} iteration(s)")

    # Resolve each pane's specs to tags that are actually present; drop empty panes.
    resolved_panes: List[Tuple[str, str, list]] = []
    for title, ylabel, specs in _build_panes(metrics):
        present = []
        for tag_name, label, color, ls in specs:
            tag = resolve_tag(metrics, tag_name, tag_name)
            if tag is not None and tag in metrics:
                present.append((tag, label, color, ls))
        if present:
            resolved_panes.append((title, ylabel, present))

    if not resolved_panes:
        print("\n⚠ No known metric families present to plot. Use scan_scalar_tags / "
              "summarize_available_tags to inspect what's there.")
        return None

    ncols = 2
    nrows = (len(resolved_panes) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4.5 * nrows), squeeze=False)
    fig.suptitle("Training Dashboard", fontsize=16)

    for idx, (title, ylabel, present) in enumerate(resolved_panes):
        ax = axes[idx // ncols][idx % ncols]
        any_plotted = False
        for tag, label, color, ls in present:
            sw = 1 if "learning_rate" in tag else smooth_window
            any_plotted |= plot_if_present(ax, metrics, tag, label, color=color,
                                           smooth_window=sw, linestyle=ls)
        _draw_iter_boundaries(ax, boundaries)
        ax.set_title(title); ax.set_xlabel("Cumulative step"); ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        if any_plotted:
            ax.legend(fontsize=9)

    # Blank any unused grid cells.
    for j in range(len(resolved_panes), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    plt.tight_layout()
    plt.show()
    print(f"\nRendered {len(resolved_panes)} pane(s) from {len(metrics)} tags.")
    return fig


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║              LIVE RUN-LEVEL LOGGER (continuous TB + W&B mirror)            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


def _finite(v) -> bool:
    try:
        return math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def _fmt(v) -> str:
    """Format a score-ish value for display ('—' for None/NaN)."""
    if v is None or not _finite(v):
        return "—"
    return f"{float(v):.3f}"


def _clip(s, n: int) -> str:
    s = "" if s is None else str(s)
    return s if len(s) <= n else s[:n] + "…"


def _tail(s, n: int) -> str:
    s = "" if s is None else str(s)
    return s if len(s) <= n else "…" + s[-n:]


def _format_samples_markdown(samples) -> str:
    """Render a few candidate records as TB-friendly markdown."""
    blocks = []
    for i, r in enumerate(samples):
        sub = r.get("sub_scores") or {}
        sub_s = ", ".join(f"Q{k}={_fmt(v)}" for k, v in sub.items())
        if r.get("pto"):
            tag = f"role=**{(r['pto'] or {}).get('role')}**"
        elif r.get("grpo"):
            g = r["grpo"] or {}
            tag = f"groupμ={_fmt(g.get('group_mean'))} σ={_fmt(g.get('group_std'))}"
        else:
            tag = ""
        la = r.get("lookahead") or {}
        la_s = ""
        if la.get("k"):
            la_s = f" · lookahead {la.get('realized_turns')}/{la.get('k')}"
        blocks.append(
            f"**#{i} — score {_fmt(r.get('score'))}** {tag}"
            f"{(' (' + sub_s + ')') if sub_s else ''}{la_s}\n\n"
            f"_…prompt tail:_ `{_tail(r.get('prompt'), 280)}`\n\n"
            f"_completion:_ {_clip(r.get('completion'), 800)}\n\n---"
        )
    return "\n\n".join(blocks)


class RunTBLogger:
    """Run-level continuous logger for a smoothable TB web UI + W&B mirror.

    TRL's per-iteration event files restart ``global_step`` at 0, so the native TB
    UI can't show a continuous cross-iteration curve (the matplotlib dashboard
    stitches them only post-hoc). This writes ONE ``SummaryWriter`` for the whole
    run to ``<local_outdir>/tb_live/`` and logs at the cumulative step, so the TB
    web UI renders continuous, smoothable curves *during* training. Custom EDA
    aggregates (mean candidate reward, oracle success, look-ahead realized turns,
    PTO τ-filter rate / GRPO group-std) + per-iteration reward histograms + a few
    sample completions land here, mirrored to W&B when active.

    All methods no-op when ``enabled=False`` or the writer failed to open, and
    every backend call is wrapped so a logging hiccup never aborts training.
    """

    def __init__(self, local_outdir: str, report_to, *, enabled: bool = True):
        self.enabled = bool(enabled)
        self.report_to = list(report_to or [])
        self.use_wandb = "wandb" in self.report_to
        self.writer = None
        self._layout_written = False
        self.log_dir = os.path.join(local_outdir, "tb_live")
        if not self.enabled:
            return
        try:
            from torch.utils.tensorboard import SummaryWriter
            os.makedirs(self.log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=self.log_dir)
            print(f"  ✓ Live TensorBoard logger → {self.log_dir}")
        except Exception as e:
            print(f"  ⚠ RunTBLogger: SummaryWriter unavailable ({e}); live TB disabled")
            self.enabled = False

    def _write_custom_layout(self) -> None:
        if self.writer is None or self._layout_written:
            return
        layout = {
            "EDA": {
                "candidate_reward": ["Multiline", ["eda/mean_candidate_reward"]],
                "oracle_success_rate": ["Multiline", ["eda/oracle_success_rate"]],
                "lookahead": ["Multiline",
                              ["eda/lookahead_realized_turns_mean", "eda/lookahead_ended_early_frac"]],
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
        try:
            self.writer.add_custom_scalars(layout)
        except Exception as e:
            print(f"  ⚠ RunTBLogger: add_custom_scalars failed ({e})")
        self._layout_written = True

    def _wandb_log(self, payload: dict, step: int) -> None:
        if not (self.use_wandb and wandb.run is not None):
            return
        try:
            wandb.log({**payload, "cumulative_global_step": int(step)})
        except Exception as e:
            print(f"  ⚠ RunTBLogger: wandb.log failed ({e})")

    def log_scalars(self, scalars: dict, *, step: int, iteration=None) -> None:
        if not self.enabled:
            return
        self._write_custom_layout()
        clean = {k: float(v) for k, v in scalars.items() if v is not None and _finite(v)}
        for k, v in clean.items():
            try:
                self.writer.add_scalar(k, v, global_step=int(step))
            except Exception as e:
                print(f"  ⚠ RunTBLogger: add_scalar({k}) failed ({e})")
        try:
            self.writer.flush()
        except Exception:
            pass
        self._wandb_log(clean, step)

    def log_histogram(self, tag: str, values, *, step: int, iteration=None) -> None:
        if not self.enabled or not values:
            return
        arr = np.asarray([float(v) for v in values if _finite(v)], dtype=float)
        if arr.size == 0:
            return
        try:
            self.writer.add_histogram(tag, arr, global_step=int(step))
            self.writer.flush()
        except Exception as e:
            print(f"  ⚠ RunTBLogger: add_histogram failed ({e})")
        if self.use_wandb and wandb.run is not None:
            try:
                wandb.log({tag: wandb.Histogram(arr), "cumulative_global_step": int(step)})
            except Exception as e:
                print(f"  ⚠ RunTBLogger: wandb histogram failed ({e})")

    def log_text(self, tag: str, text: str, *, step: int) -> None:
        if not self.enabled:
            return
        try:
            self.writer.add_text(tag, text, global_step=int(step))
            self.writer.flush()
        except Exception as e:
            print(f"  ⚠ RunTBLogger: add_text failed ({e})")

    def log_sample_completions(self, samples, *, step: int, iteration=None) -> None:
        """Log a few candidate records as TB markdown text + a W&B table.

        Replaces the noisy inline GRPO completion dump: a readable spread of
        best/median/worst completions you can browse in the TB UI / W&B.
        """
        if not self.enabled or not samples:
            return
        tag = f"samples/iteration_{iteration}" if iteration is not None else "samples"
        self.log_text(tag, _format_samples_markdown(samples), step=step)
        if self.use_wandb and wandb.run is not None:
            try:
                cols = ["score", "tag", "prompt_tail", "completion"]
                rows = []
                for r in samples:
                    if r.get("pto"):
                        t = (r["pto"] or {}).get("role")
                    elif r.get("grpo"):
                        g = r["grpo"] or {}
                        t = f"gμ={_fmt(g.get('group_mean'))}/σ={_fmt(g.get('group_std'))}"
                    else:
                        t = ""
                    rows.append([_fmt(r.get("score")), t,
                                 _tail(r.get("prompt"), 280), _clip(r.get("completion"), 800)])
                wandb.log({f"samples/iter_{iteration}": wandb.Table(columns=cols, data=rows),
                           "cumulative_global_step": int(step)})
            except Exception as e:
                print(f"  ⚠ RunTBLogger: wandb table failed ({e})")

    def close(self) -> None:
        if self.writer is not None:
            try:
                self.writer.flush()
                self.writer.close()
            except Exception:
                pass
            self.writer = None
