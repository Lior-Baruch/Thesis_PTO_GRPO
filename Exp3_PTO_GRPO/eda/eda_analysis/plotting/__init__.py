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
- :mod:`.behavior`      — the generic behaviour-count detail grid (MITI/MICI/PCT + session shape),
  MITI 4.2.1 thresholds, question-rate cross-check.
- :mod:`.questionnaires` — per-questionnaire item drill-down: the uniform item trajectory grid +
  the "which items drive the change" delta bars (+ the Q2 face-content specializations).
- :mod:`.training`      — TRAINING-signal figures (reward distributions, advantage side-by-side).
- :mod:`.reliability`   — MEASUREMENT-validity figures (oracle ICC, second-judge agreement +
  contrast preservation), from the ``data/eval_scores/judge=<tag>/rep=<r>/`` score lake.
- :mod:`.compute`       — the COMPUTE axis: score vs cumulative GPU-hours, the budget sweep
  (does the lever pay at equal spend?), and where each arm's hours go.
- :mod:`.lookahead`     — RQ-i: the BEHAVIOUR channels the K=0-vs-K=5 reward curve hides
  (channel trajectories, the select→generate→evaluate mechanism panel, the trade-off forest) +
  the promoted paper figures (four-arm headline, delta grids, channels grid, both-judges contrast,
  retention by K, DiD).
- :mod:`.tails`         — the look-ahead TAIL audit (ended-early share, within-group deviation,
  P(argmax) by tail state) + API calls per iteration.
- :mod:`.dispersion`    — within-group reward dispersion by K (SD / margin / margin-over-SD /
  winner-z) + the τ-yield sensitivity panel.
- :mod:`.faithfulness`  — reward faithfulness by K (rank-agreement curves + matched-policy bins),
  one figure per eval grader.
- :mod:`.crossgen`      — Exp1 (ICLR) under two graders (wide + column variants).
- :mod:`.replication`   — session-shape trajectories + score-SD stability panels.
- :mod:`.instruments`   — held-out instruments under K: WAI-SR subscale gains, heterogeneity grid.

``K_STYLE`` (K=0 solid+circle / K=5 dashed+square) is defined by EACH promoted plotting module
(``plotting.lookahead.K_STYLE``, ``plotting.tails.K_STYLE``, … — ``plotting.instruments.K_STYLE``
also carries a ``hatch`` key) and is deliberately NOT re-exported at this package level: eight
same-named module constants would make ``plotting.K_STYLE`` mean one of several. Qualify it.

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
)
from .questionnaires import (  # noqa: F401
    item_trajectory_grid, item_delta_bars, q2_item_delta_bars, q2_item_group_trajectory,
)
from .training import (  # noqa: F401
    reward_distribution, advantage_signal_sidebyside,
)
from .reliability import (  # noqa: F401
    oracle_repeatability_bars, judge_agreement_scatter, judge_contrast_bars,
    judge_dumbbell, variance_decomposition_bars, gain_retention_bars, concordance_curve,
    retention_trajectory,
)
from .lookahead import (  # noqa: F401
    k_channel_trajectory, k_channel_trajectory_grid, k_mechanism_panel, k_channel_forest,
    k_cost_benefit,
    # promoted paper figures (k_contrast_headline + cross_k_multijudge)
    k_headline_fourarm, k_delta_grid, k_channels_grid, k_contrast_both_judges, k_retention, k_did,
)
from .compute import (  # noqa: F401
    compute_trajectory, budget_sweep_plot, cost_breakdown,
    # promoted paper figures (compute_axis)
    BREAKDOWN_NOTES, cost_breakdown_by_arm, cost_breakdown_by_iteration, budget_sweep_grid,
    trajectory_by_compute,
)
from .tails import tail_audit_fig, api_calls_fig                       # noqa: F401
from .dispersion import dispersion_fig, tau_fig                        # noqa: F401
from .faithfulness import faithfulness_fig                             # noqa: F401
from .crossgen import crossgen_fig                                     # noqa: F401
from .replication import shape_fig, sd_fig                             # noqa: F401
from .instruments import wai_fig, hetero_fig                           # noqa: F401
# The promoted modules are also reachable as submodules (``plotting.lookahead.K_STYLE`` etc.).
from . import (lookahead, compute, tails, dispersion, faithfulness, crossgen,   # noqa: F401
               replication, instruments)

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
    # questionnaires
    "item_trajectory_grid", "item_delta_bars", "q2_item_delta_bars", "q2_item_group_trajectory",
    # training
    "reward_distribution", "advantage_signal_sidebyside",
    # reliability (measurement validity)
    "oracle_repeatability_bars", "judge_agreement_scatter", "judge_contrast_bars",
    "judge_dumbbell", "variance_decomposition_bars", "gain_retention_bars", "concordance_curve",
    "retention_trajectory",
    # lookahead (RQ-i: the behaviour channels the reward curve hides + the promoted paper figures)
    "k_channel_trajectory", "k_channel_trajectory_grid", "k_mechanism_panel", "k_channel_forest",
    "k_cost_benefit",
    "k_headline_fourarm", "k_delta_grid", "k_channels_grid", "k_contrast_both_judges",
    "k_retention", "k_did",
    # compute (the GPU-hour axis: score vs budget, the budget sweep, the cost breakdown)
    "compute_trajectory", "budget_sweep_plot", "cost_breakdown",
    "BREAKDOWN_NOTES", "cost_breakdown_by_arm", "cost_breakdown_by_iteration", "budget_sweep_grid",
    "trajectory_by_compute",
    # tails (look-ahead tail audit + API calls)
    "tail_audit_fig", "api_calls_fig",
    # dispersion (within-group reward dispersion by K + tau sensitivity)
    "dispersion_fig", "tau_fig",
    # faithfulness (reward faithfulness by K, one figure per grader)
    "faithfulness_fig",
    # crossgen (Exp1 under two graders)
    "crossgen_fig",
    # replication (session shape + score-SD stability)
    "shape_fig", "sd_fig",
    # instruments (WAI-SR subscale gains, heterogeneity grid)
    "wai_fig", "hetero_fig",
    # the promoted plotting submodules (K_STYLE lives on each; not re-exported here — see docstring)
    "lookahead", "compute", "tails", "dispersion", "faithfulness", "crossgen", "replication",
    "instruments",
]
