# analysis/ — the paper's frozen fixture

**Nothing in this folder is regenerated.** It is the record of what the paper's numbers were
computed from, kept so the text can be audited against a fixed artifact and so the EDA can prove it
reproduces it.

## What is here

| Path | What it is |
|---|---|
| `out/<generator>.json` | one ledger per retired generator (`k_contrast_headline`, `cross_k_multijudge`, `compute_axis`, `tail_audit`, `session_shape_stability`, `crossgen_exp1`, `held_out_instruments`, `dispersion_by_k`, `reward_faithfulness`): every quotable number with its source table, bootstrap CIs at **seed 0** |
| `out/_findings_digest.txt` | the generators' findings / caveats / paper-use notes — the "digest" and "caveats" entries `NUMBERS.md` cites |
| `out/_appendix_rows.tex` | the appendix table rows the generators emitted |
| `../tables/*.md\|csv` | the 176 tables the generators wrote (same fixture; kept beside the `.tex` because `NUMBERS.md` names them) |

The EDA self-check (`eda_analysis._selfcheck`, "paper fixture anchors") reads `out/*.json` to
assert that the promoted modules reproduce the fixture: every mean, dz, p and count is identical;
bootstrap CI bounds may differ in the third decimal because the EDA bootstraps at
`BOOT_SEED=12345`.

## Where the code went

The nine generators (`analysis/*.py` + `_common.py`) were retired on 2026-08-18. They live in git
at commit `b09eb6f` (the last pre-reorg state) and were promoted, one module each, into the tracked
EDA package `Exp3_PTO_GRPO/eda/eda_analysis/`:

| Retired generator | Promoted to | Renders into `eda/results/…` |
|---|---|---|
| `k_contrast_headline.py` | `eda_analysis.lookahead` (+ channels in `lookahead/behaviour`) | `lookahead/reward`, `lookahead/behaviour` |
| `cross_k_multijudge.py` | `eda_analysis.transfer` (pairs, ladder, retention) + `eda_analysis.lookahead` (DiD, method gap, endpoints) | `lookahead/transfer`, `lookahead/reward` |
| `compute_axis.py` | `eda_analysis.compute` | `compute/cost` |
| `tail_audit.py` | `eda_analysis.tails` (API tables in `compute`) | `lookahead/mechanism`, `compute/cost` |
| `session_shape_stability.py` | `eda_analysis.replication` (+ shape/length/selection in `lookahead/behaviour`) | `lookahead/replication`, `lookahead/behaviour` |
| `crossgen_exp1.py` | `eda_analysis.crossgen` | `lookahead/replication` |
| `held_out_instruments.py` | `eda_analysis.instruments` | `lookahead/behaviour` |
| `dispersion_by_k.py` | `eda_analysis.dispersion` | `lookahead/mechanism` |
| `reward_faithfulness.py` | `eda_analysis.faithfulness` | `lookahead/mechanism` |

Regenerate the live artifacts with
`.venv\Scripts\python.exe Exp3_PTO_GRPO\eda\tools\render_results.py --top lookahead compute`, then
`sync_figures.py` (see `../README.md` § Regenerating). The fixture-table → results-table map is in
`../NUMBERS.md` (each source cell names the tracked successor first and the fixture in parentheses).
