"""_shared.py — tiny helpers used by more than one plotting submodule (leaf within the subpackage)."""

from typing import Optional, Sequence

from ..constants import QUESTIONNAIRE_ORDER, k_of  # noqa: F401  (k_of re-exported for the twins)

# Okabe-Ito qualitative colors for nominal categories (persona traits, Q2 item groups) —
# distinct from the arm palette in plotting_style.
_QUAL_COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]

# THE K encoding: K=0 solid + circle, K=5 dashed + square. Every caption in `results/` states it,
# so it is a cross-figure reading convention, not a per-module style choice.
#
# ⚠ One definition on purpose. It was copied into eight plotting modules; changing the convention
# meant editing eight files, and missing one made two figures in the same paper disagree about
# which line is K=5. Each module still does `from ._shared import K_STYLE`, so the qualified name
# (`plotting.tails.K_STYLE`) and every ``__all__`` entry keep resolving — which is what
# `plotting/__init__.py` asks for by declining to re-export an ambiguous flat `plotting.K_STYLE`.
K_STYLE = {0: {"ls": "-", "marker": "o"}, 5: {"ls": "--", "marker": "s"}}

# The bar/area twin adds a greyscale-safe hatch on the same two keys.
K_STYLE_HATCHED = {k: {**v, "hatch": h} for (k, v), h in zip(K_STYLE.items(), ("", "//"))}


def _metrics(frame_metrics, metrics: Optional[Sequence[str]]) -> list:
    """The requested metrics (default = canonical order), filtered to those present in the frame."""
    present = set(frame_metrics)
    return [m for m in (metrics or QUESTIONNAIRE_ORDER) if m in present]
