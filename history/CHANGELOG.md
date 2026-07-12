# Root CHANGELOG — Thesis_PTO_GRPO

**A thin cross-experiment index, one line per landed change, newest first.** The detailed dated
narratives live in each experiment's own changelog — today that is
[Exp3_PTO_GRPO/history/CHANGELOG.md](../Exp3_PTO_GRPO/history/CHANGELOG.md) (Exp1 is frozen, Exp2
complete, so every entry below points there). Converted from parallel prose to this index on
2026-07-12 (docs refactor: one owner per fact); root-only details were merged into the Exp3 changelog
first — nothing was deleted.

---

- **2026-07-12** — Docs refactor: one owner per fact (Exp3 CLAUDE.md 601→492; this file → index; eda README/METRICS numbers → pointers; root CLAUDE.md "Doc map" + consolidated "Current status").
- **2026-07-11** — EDA roadmap #7: Run_Eval's `EXPERIMENTS` registry auto-generated from `discover_arms()`; empty `model_iter` dirs skipped; LA5 pause-state docs corrected; backlog clear.
- **2026-07-08** — **GRPO LA0 FINISHED (10 iters) + re-scored: PTO wins the matched endpoint (Q1+Q2 4.26 vs 3.75); GRPO peaks @8 then regresses into sycophancy.** Same day: 20-commit EDA hardening/refactor (parquet cache, `_selfcheck`, constants leaf, output-clean notebooks).
- **2026-07-07** — EDA backlog #7 (general review) done: MI-SAT domain bug fixed + all 2,784 convs re-scored (~+0.14 uniform, no headline change); honest PTO-vs-GRPO advantage signal (branch_id grouping bug caught); MITI rate-normalization; framing pass + LIMITATIONS.md.
- **2026-07-03** — EDA backlog #1–#6 cleared (grid+subfolder everywhere, GRPO `group_range` analog, question syntax-vs-function resolved, labels, warmth-vs-orthogonal explainer, stats audit — no correctness bugs).
- **2026-07-02** — EDA reorg-by-topic (notebooks ↔ numbered result families 1:1) + reward-hacking figures + readable-labels layer; 7-item backlog opened.
- **2026-06-18** — VIEW system (`all`/`L0`/`L5` trees + `render_views.py`) + package consolidation 14→9 modules + hand-authored per-view SUMMARY.md.
- **2026-06-14** — Orthogonal eval axes (PCT + MICI + derived R:Q/%CR/%MICO; PC1 91%→≈56%) + `EdaConfig` control surface + PNG/xlsx exports (2 passes).
- **2026-06-10** — EDA restructured (3 passes: by research question → readability fixes → by purpose, `[EVAL]`/`[TRAINING]` tags, effect forest, TB curves, Exp3 reliability-curve rebuild).
- **2026-06-09** — EDA rebuilt research-grade (`eda_analysis/` package, true-persona recovery, paired stats) + first cross-method results (early-GRPO snapshot, later superseded).
- **2026-06-08** — Sub-epoch checkpointing (`SAVE_STEPS=10`) + walk-back resume + EDA-completeness-on-resume (both trainers).
- **2026-06-07** — Four landings: ChatML self-play/role-swap leak fixed (stop-strings + clean + reward floor); both quicktests PASSED end-to-end locally (3-agent review: no bugs); Colab throughput tuning (3 arms launched); PTO Step-2 pref-build auto-resume.
- **2026-06-06/07** — First full Colab runs diagnosed + fixed: PTO DPO OOM (prompt cap + 2×8 + grad-ckpt) and GRPO length reward-hack (`stop_strings`); logging reverted to HF defaults.
- **2026-06-05** — Per-generation EDA capture (`iteration_N/eda/generations.jsonl`) + opt-in live TensorBoard.
- **2026-06-04 (and before)** — Batched lock-step look-ahead (GPU-validated, |Δmean|=0.024) + PTO parity with GRPO + greedy true-PTO mode + training oracle in `EXPERIMENT_NAME` + iter-2 local-crash fix (`precompute_ref_log_probs`); torchao Colab crash fixed.
- **2026-06-01/03** — Dependency-stack audit (pinned stack verified current; torchao uninstall; batch/iteration bumps).
