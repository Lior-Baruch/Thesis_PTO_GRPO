"""
eda_analysis.plotting — the figure layer: the named recurring plot functions.

Split into topic submodules (2026-07-13; formerly one 935-line ``plotting.py``) — this
``__init__`` re-exports everything, so the public surface is unchanged
(``eda_analysis.plotting.<figure>`` and the ``figures``/``plots`` aliases keep resolving):

- :mod:`.outcomes`      — per-model bars, the vs-base effect forest, the leaderboard scorecard.
- :mod:`.trajectories`  — per-rubric trajectory grid, single-metric curves (peak flagging),
  WAI/MITI subscale grids, the reward-hack twin-axis panel.
- :mod:`.heterogeneity` — persona-trait splits (per-arm grid, all-metric overview, endpoint bars).
- :mod:`.structure`     — reward-faithfulness (reliability curve, proxy-vs-eval) + rubric
  structure (correlation heatmap, factor-loadings bars).
- :mod:`.behavior`      — behaviour drift, MITI 4.2.1 thresholds, question-rate cross-check,
  Q2 item-level reward composition.
- :mod:`.training`      — TRAINING-signal figures (reward distributions, advantage side-by-side).

The style/scaffold helpers live in :mod:`eda_analysis.plotting_style` and are re-imported here so
``figures.set_style(...)`` / ``figures.grid(...)`` etc. still resolve on this package.

Contract for every named-plot function: takes an already-built tidy frame (never touches disk),
returns a matplotlib ``fig`` (no ``plt.show()`` / ``save_fig`` — the notebook owns those), reuses
the ``plotting_style`` helpers, and degrades gracefully on thin/absent arms (returns ``None`` or
an empty panel).
"""

# Style/scaffold helpers — re-exported so this package (and its ``figures``/``plots`` aliases)
# exposes set_style/arm_palette/grid/... exactly as the flat module did.
from ..plotting_style import (  # noqa: F401
    set_style, arm_palette, apply_score_axis, model_order, clean_label,
    relabel_xticks, relabel_legend, add_base_line, figure_legend_from, grid,
)

from .outcomes import (  # noqa: F401
    outcomes_by_model, effect_forest, leaderboard_scorecard,
)
from .trajectories import (  # noqa: F401
    trajectory_grid, single_metric_trajectory, subscale_trajectory_grid, reward_hack_panel,
)
from .heterogeneity import (  # noqa: F401
    heterogeneity_grid, heterogeneity_overview_grid, subgroup_endpoint_bars,
)
from .structure import (  # noqa: F401
    reliability_curve, faithfulness_proxy_vs_eval, rubric_correlation_heatmap,
    factor_loadings_bars,
)
from .behavior import (  # noqa: F401
    behavior_trajectory_grid, single_behavior_trajectory,
    miti_threshold_panel, miti_threshold_table, question_rate_crosscheck,
    q2_item_delta_bars, q2_item_group_trajectory,
)
from .training import (  # noqa: F401
    reward_distribution, advantage_signal_sidebyside,
)

__all__ = [
    # style helpers (from plotting_style)
    "set_style", "arm_palette", "apply_score_axis", "model_order", "clean_label",
    "relabel_xticks", "relabel_legend", "add_base_line", "figure_legend_from", "grid",
    # outcomes
    "outcomes_by_model", "effect_forest", "leaderboard_scorecard",
    # trajectories
    "trajectory_grid", "single_metric_trajectory", "subscale_trajectory_grid", "reward_hack_panel",
    # heterogeneity
    "heterogeneity_grid", "heterogeneity_overview_grid", "subgroup_endpoint_bars",
    # structure
    "reliability_curve", "faithfulness_proxy_vs_eval", "rubric_correlation_heatmap",
    "factor_loadings_bars",
    # behavior
    "behavior_trajectory_grid", "single_behavior_trajectory",
    "miti_threshold_panel", "miti_threshold_table", "question_rate_crosscheck",
    "q2_item_delta_bars", "q2_item_group_trajectory",
    # training
    "reward_distribution", "advantage_signal_sidebyside",
]
