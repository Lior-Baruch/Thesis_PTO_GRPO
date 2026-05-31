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
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
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


def plot_iteration_metrics(log_root, smooth_window: int = 2):
    """Render the standard 2×2 training dashboard: loss / reward / KL+entropy / LR.

    Args:
        log_root: directory containing ``events.out.tfevents.*`` files (recursed).
        smooth_window: moving-average window for visual smoothing only.
    """
    print("Parsing TensorBoard event files...")
    metrics = parse_tensorboard_logs(log_root)
    boundaries = compute_iteration_boundaries(log_root)

    if not metrics:
        print("\n⚠ No metrics found in event files. Check if training has completed.")
        return None

    print(f"Found {len(metrics)} scalar tags across {len(boundaries)} iteration(s)")

    tags = {
        "train_loss": resolve_tag(metrics, "train/loss", "train/loss"),
        "eval_loss": resolve_tag(metrics, "eval/loss", "eval/loss"),
        "train_reward": resolve_tag(metrics, "train/reward", "train/reward"),
        "eval_reward": resolve_tag(metrics, "eval/reward", "eval/reward"),
        "train_reward_std": resolve_tag(metrics, "train/reward_std", "train/reward_std"),
        "eval_reward_std": resolve_tag(metrics, "eval/reward_std", "eval/reward_std"),
        "train_kl": resolve_tag(metrics, "train/kl", "train/kl"),
        "eval_kl": resolve_tag(metrics, "eval/kl", "eval/kl"),
        "train_entropy": resolve_tag(metrics, "train/entropy", "train/entropy"),
        "eval_entropy": resolve_tag(metrics, "eval/entropy", "eval/entropy"),
        "lr": resolve_tag(metrics, "train/learning_rate", "learning_rate"),
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Training Dashboard (Key Metrics)", fontsize=16)

    # 1) Loss
    ax = axes[0, 0]
    found = False
    found |= plot_if_present(ax, metrics, tags["train_loss"], "Train loss", color="#1f77b4", smooth_window=smooth_window)
    found |= plot_if_present(ax, metrics, tags["eval_loss"], "Eval loss", color="#ff7f0e", smooth_window=smooth_window)
    _draw_iter_boundaries(ax, boundaries)
    ax.set_title("Loss"); ax.set_xlabel("Cumulative step"); ax.set_ylabel("Loss"); ax.grid(True, alpha=0.3)
    (ax.legend(fontsize=9) if found else _annotate_empty(ax, "No loss metrics found"))

    # 2) Reward
    ax = axes[0, 1]
    found = False
    found |= plot_if_present(ax, metrics, tags["train_reward"], "Train reward", color="#2ca02c", smooth_window=smooth_window)
    found |= plot_if_present(ax, metrics, tags["eval_reward"], "Eval reward", color="#d62728", smooth_window=smooth_window)
    found |= plot_if_present(ax, metrics, tags["train_reward_std"], "Train reward std", color="#9467bd", smooth_window=smooth_window, linestyle="--")
    found |= plot_if_present(ax, metrics, tags["eval_reward_std"], "Eval reward std", color="#8c564b", smooth_window=smooth_window, linestyle="--")
    _draw_iter_boundaries(ax, boundaries)
    ax.set_title("Reward"); ax.set_xlabel("Cumulative step"); ax.set_ylabel("Reward"); ax.grid(True, alpha=0.3)
    (ax.legend(fontsize=9) if found else _annotate_empty(ax, "No reward metrics found"))

    # 3) KL + Entropy
    ax = axes[1, 0]
    found = False
    found |= plot_if_present(ax, metrics, tags["train_kl"], "Train KL", color="#17becf", smooth_window=smooth_window)
    found |= plot_if_present(ax, metrics, tags["eval_kl"], "Eval KL", color="#bcbd22", smooth_window=smooth_window)
    found |= plot_if_present(ax, metrics, tags["train_entropy"], "Train entropy", color="#7f7f7f", smooth_window=smooth_window, linestyle="--")
    found |= plot_if_present(ax, metrics, tags["eval_entropy"], "Eval entropy", color="#e377c2", smooth_window=smooth_window, linestyle="--")
    _draw_iter_boundaries(ax, boundaries)
    ax.set_title("KL / Entropy"); ax.set_xlabel("Cumulative step"); ax.set_ylabel("Value"); ax.grid(True, alpha=0.3)
    (ax.legend(fontsize=9) if found else _annotate_empty(ax, "No KL/entropy metrics found"))

    # 4) Learning rate
    ax = axes[1, 1]
    found = plot_if_present(ax, metrics, tags["lr"], "Learning rate", color="#1f77b4", smooth_window=1)
    _draw_iter_boundaries(ax, boundaries)
    ax.set_title("Learning Rate"); ax.set_xlabel("Cumulative step"); ax.set_ylabel("LR"); ax.grid(True, alpha=0.3)
    (ax.legend(fontsize=9) if found else _annotate_empty(ax, "No learning-rate metric found"))

    plt.tight_layout()
    plt.show()

    print("\nKey tags used:")
    for k, v in tags.items():
        print(f"  {k}: {v}")
    return fig
