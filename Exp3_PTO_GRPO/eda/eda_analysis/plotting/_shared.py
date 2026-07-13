"""_shared.py — tiny helpers used by more than one plotting submodule (leaf within the subpackage)."""

from typing import Optional, Sequence

from ..constants import QUESTIONNAIRE_ORDER

# Okabe-Ito qualitative colors for nominal categories (persona traits, Q2 item groups) —
# distinct from the arm palette in plotting_style.
_QUAL_COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]


def _metrics(frame_metrics, metrics: Optional[Sequence[str]]) -> list:
    """The requested metrics (default = canonical order), filtered to those present in the frame."""
    present = set(frame_metrics)
    return [m for m in (metrics or QUESTIONNAIRE_ORDER) if m in present]
