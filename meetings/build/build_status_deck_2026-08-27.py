#!/usr/bin/env python
"""
build_status_deck_2026-08-27.py — the 2026-08-27 supervisor deck.

THE COMPLETE-GRID DECK. The first built after all four arms reached iteration 10 (GRPO K=5's
adapter landed 2026-08-25) and after the replicate draw confirmed every headline. Spine:

    ACT I    Exp3 results   — the four arms at the endpoint; the K x optimizer interaction;
                              the reward-hack asymmetry; measurement honesty; the compute axis.
    ACT II   Papers         — P1 (ICLR 2027, GRPO-only) and P2 (ARR Oct, the full 2x2).
    ACT III  Exp4           — why the open-stack side project exists and what it unlocks.

The organizing claim, stated on slide 12: look-ahead is a PERFORMANCE lever for GRPO (best
endpoint on both graders + over-praise capture suppressed ~10x) and only a HYGIENE lever for PTO
(no reward gain at any iteration on either grader, but over-praise capture suppressed ~5x).

NUMBERS ARE HARD-CODED, like every builder here: a deck is a snapshot of what was presented on
its date, not a live view. Every number below was verified against its owning table on
2026-08-27, reading the tables directly (not STATUS.md prose). Owning artifacts, under
``Exp3_PTO_GRPO/eda/results/`` unless marked otherwise:

    method/contrast/tables/method_paired_by_K.md            PTO - GRPO at matched K + iteration
    arms/outcomes/tables/<judge>/leaderboard_scorecard.md   endpoint levels, both graders
    lookahead/reward/tables/k_table1.md                     K contrast by iteration (+ = K0 higher)
    lookahead/reward/tables/k_did.md                        the K x method difference-in-differences
    arms/validity/tables/gpt-4o-mini/overpraise_crosscheck.md   lexical + rated over-praise rates
    measurement/validity/tables/judge_saturation_data.md    cross-judge agreement + SD by iteration
    measurement/replicate_draw.md                           the second independent draw
    compute/cost/tables/compute_by_arm.md                   GPU-hours per arm
    compute/cost/tables/budget_sweep_crossjudge_verdicts.md matched-budget verdicts
    Exp4_OpenStack/CLAUDE.md                                the Exp4 spec + gate status

Build:
    & ..\\..\\.venv\\Scripts\\python.exe build_status_deck_2026-08-27.py
    .\\export_pdf.ps1 ..\\2026-08-27\\status_2026-08-27.pptx
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from _deck_kit import (  # noqa: E402
    BODY, CAVEAT, CW, CWASH, Deck, DUSK, GOLD, GRPO_C, H, INK, MINT, MIST, ML, MUTED, PANEL,
    PAPER, PTO_C, SKY, SLATE, VERDICT, VWASH, W, WASH,
    band, bandbot, bullets, caption, figpage, para, pic, provenance, rect, run, table,
    table_bottom, txbox,
)
from pptx.enum.text import PP_ALIGN  # noqa: E402
from pptx.util import Inches  # noqa: E402

REPO = os.path.dirname(os.path.dirname(HERE))
ROOT = os.path.join(REPO, "Exp3_PTO_GRPO")
RES = os.path.join(ROOT, "eda", "results")
OUTDIR = os.path.join(REPO, "meetings", "2026-08-27")
OUT = os.path.join(OUTDIR, "status_2026-08-27.pptx")


def fig(*parts):
    return os.path.join(RES, *parts)


d = Deck()


# ══════════════════════════════════════════════════════════════════════════════
# 1 · TITLE
# ══════════════════════════════════════════════════════════════════════════════
s = d._next()
rect(s, 0, 0, W, H, INK)
rect(s, ML, Inches(1.72), Inches(1.5), Inches(0.05), GRPO_C)
tf = txbox(s, ML, Inches(2.05), CW - Inches(1.0), Inches(2.9))
p = para(tf, first=True)
run(p, "The grid is complete.", size=44, bold=True, color=PAPER)
p = para(tf, space_before=2)
run(p, "Four arms, ten iterations, two graders —", size=30, bold=True, color=SKY)
p = para(tf, space_before=1)
run(p, "and the answers interact.", size=30, bold=True, color=SKY)
p = para(tf, space_before=18)
run(p, "Exp3 endpoint results   ·   two paper drafts   ·   Exp4 built and gated", size=16,
    color=MIST)
tf = txbox(s, ML, H - Inches(1.66), CW, Inches(1.1))
p = para(tf, first=True)
run(p, "Lior Baruch   ·   Reichman University   ·   27 August 2026", size=13, color=SLATE)
p = para(tf, space_before=6)
run(p, "Llama-3.2-1B therapist  ·  gpt-4o-mini patient + training oracle  ·  "
       "Claude Haiku 4.5 held-out judge  ·  96 personas  ·  8 instruments", size=10.5, color=DUSK)


# ══════════════════════════════════════════════════════════════════════════════
# 2 · WHERE THINGS STAND
# ══════════════════════════════════════════════════════════════════════════════
s, y = d.newslide("Status", "All four arms complete — Exp3 is data-complete")
rows = [
    ("PTO  K=0", "1–10", "Base + I1–I10  =  11", "8.119"),
    ("PTO  K=5", "1–10", "Base + I1–I10  =  11", "19.681"),
    ("GRPO  K=0", "1–10", "Base + I1–I10  =  11", "27.906"),
    ("GRPO  K=5", "1–10", "Base + I1–I10  =  11", "51.205"),
]
table(s, y + Inches(0.05),
      ["Arm", "Iterations trained", "Model states scored (both graders)", "GPU-hours"],
      rows, col_w=[Inches(2.4), Inches(2.6), Inches(4.2), Inches(2.0)], size=13,
      row_h=Inches(0.44))
y2 = table_bottom(y + Inches(0.05), len(rows), row_h=Inches(0.44))
y2 = caption(s, y2, "44 model states  x  8 instruments  x  96 personas  =  33,792 scored cells "
                    "per grader. Nothing is waiting to be scored; the matched iteration-10 "
                    "endpoint every headline needed exists.")
y2 = band(s, y2 + Inches(0.15), "WHAT THIS MEANS",
          "Training and scoring are finished. Everything from here is analysis and write-up — "
          "no further training spend in Exp3.",
          fill=VWASH, edge=VERDICT, label_color=VERDICT)
provenance(s, ["STATUS.md", "compute/cost/tables/compute_by_arm.md"])


# ══════════════════════════════════════════════════════════════════════════════
# 3 · THE DESIGN (reminder)
# ══════════════════════════════════════════════════════════════════════════════
s, y = d.newslide("Reminder", "One 2x2: optimizer family  x  reward horizon")
bullets(s, y + Inches(0.02), [
    ("PTO vs GRPO.", "Preference trees + DPO loss vs group-relative RL. Both iterative, sharing "
     "the same generation, oracle and look-ahead machinery; matched MCL=12, M = G = 8, "
     "temperatures, 96 patient personas per iteration."),
    ("K-turn look-ahead.", "The oracle scores the K-turn continuation a candidate turn leads to, "
     "not the turn itself. K in {0, 5}, the same lever in both methods — only what the oracle "
     "sees changes, never the loss."),
    ("Reward vs eval.", "Training reward is Q1+Q2 only (as in the ICLR paper). Evaluation is all "
     "8 instruments: the 6 MI questionnaires + patient change-talk (PCT) + MI-inconsistency "
     "(MICI, lower = better)."),
    ("Two graders.", "gpt-4o-mini is the training oracle; Claude Haiku 4.5 is held out — a "
     "different model family that never played the patient and was never the reward. Levels are "
     "never compared across graders."),
], size=12.5, gap=7)
pic(s, fig("schematics", "pto_framework.png"), ML, Inches(4.55), Inches(5.7), Inches(2.15))
pic(s, fig("schematics", "grpo_framework.png"), ML + Inches(5.95), Inches(4.55), Inches(5.7),
    Inches(2.15))
provenance(s, ["schematics/pto_framework.png", "schematics/grpo_framework.png"])


# ══════════════════════════════════════════════════════════════════════════════
# 4 · DIVIDER — ACT I
# ══════════════════════════════════════════════════════════════════════════════
d.divider("ACT I  ·  EXP3 RESULTS",
          "Four arms at the endpoint",
          "Each research question now has an answer — and none of them is unconditional.",
          accent=GRPO_C)


# ══════════════════════════════════════════════════════════════════════════════
# 5 · HEADLINE GRID (figure)
# ══════════════════════════════════════════════════════════════════════════════
figpage(d, "Exp3 · the four arms",
        "Where the four arms land — every instrument, both graders, each arm vs its own base",
        fig("method", "contrast", "figures", "headline_grid.png"),
        [("WHAT", "Endpoint (iteration 10) change vs each arm's own base, per instrument, with "
                  "persona-bootstrap CIs."),
         ("GRADERS", "Primary and held-out side by side on independent axes — levels are never "
                     "compared across graders."),
         ("N", "96 personas per model state; persona-paired throughout."),
         ("READ", "GRPO K=5 leads on the primary and holds on the held-out; GRPO K=0 is the weak "
                  "arm on both.")],
        ["method/contrast/figures/headline_grid.png", "method/contrast/tables/headline_grid.md"])


# ══════════════════════════════════════════════════════════════════════════════
# 6 · THE METHOD VERDICT FLIPS WITH K
# ══════════════════════════════════════════════════════════════════════════════
s, y = d.newslide("Exp3 · RQ-ii  (PTO vs GRPO)",
                  "PTO vs GRPO: the verdict flips sign with K")
rows = [
    ("K = 0", "+0.507   (dz 0.729) ***", "+0.609   (dz 1.265) ***", "PTO wins"),
    ("K = 5", "−0.210   (dz −0.356) **", "−0.206   (dz −0.313) *", "GRPO wins"),
]
table(s, y + Inches(0.1),
      ["PTO − GRPO,  Q1+Q2 @ iteration 10", "primary (gpt-4o-mini)",
       "held-out (Claude Haiku 4.5)", "verdict"],
      rows, col_w=[Inches(3.5), Inches(3.1), Inches(3.1), Inches(1.7)], size=13,
      row_h=Inches(0.5), emphasis=lambda i, j, v: j == 3)
y2 = table_bottom(y + Inches(0.1), len(rows), row_h=Inches(0.5))
y2 = caption(s, y2, "Sign: + = PTO higher. n = 96, persona-paired, Holm-corrected within grader. "
                    "* p<.05   ** p<.01   *** p<.001.")
y2 = band(s, y2 + Inches(0.12), "VERDICT",
          "The method comparison is an interaction with the reward horizon: PTO wins at K=0, "
          "GRPO wins at K=5 — significant on both graders, at the same iteration, on the same "
          "conversations.",
          fill=VWASH, edge=VERDICT, label_color=VERDICT)
bandbot(s, "NOT THIS",
        "“PTO beats GRPO” or “GRPO beats PTO”, unconditioned. Neither "
        "survives the table — a method claim without naming K is wrong at the other K.",
        fill=CWASH, edge=CAVEAT, label_color=CAVEAT, size=12.5)
provenance(s, ["method/contrast/tables/method_paired_by_K.md"])


# ══════════════════════════════════════════════════════════════════════════════
# 7 · K LEVER TRAJECTORIES (figure)
# ══════════════════════════════════════════════════════════════════════════════
figpage(d, "Exp3 · RQ-i  (look-ahead)",
        "The look-ahead lever by iteration — all four arms, both graders",
        fig("lookahead", "reward", "figures", "k_headline_q1q2.png"),
        [("SERIES", "Q1+Q2 level curves for all four arms, base through iteration 10, plus the "
                    "persona-paired K contrast strip."),
         ("GRADERS", "Both, independent axes."),
         ("READ", "The GRPO pair separates in K=5's favour from iteration ~4 and the gap is "
                  "largest at the endpoint; the PTO pair never separates."),
         ("N", "96 personas per state, persona-paired.")],
        ["lookahead/reward/figures/k_headline_q1q2.png", "lookahead/reward/tables/k_table1.md"])


# ══════════════════════════════════════════════════════════════════════════════
# 8 · LOOK-AHEAD VERDICT
# ══════════════════════════════════════════════════════════════════════════════
s, y = d.newslide("Exp3 · RQ-i  (look-ahead)",
                  "Look-ahead pays for GRPO — and not for PTO's reward")
rows = [
    ("GRPO", "−0.765   (dz −0.91) ***", "−0.616   (dz −1.03) ***", "K=5 clearly ahead"),
    ("PTO", "−0.047   (dz −0.10)  n.s.", "+0.199   (dz +0.31)  n.s.", "null"),
]
table(s, y + Inches(0.1),
      ["K=0 − K=5,  Q1+Q2 @ iteration 10", "primary", "held-out", "verdict"],
      rows, col_w=[Inches(3.5), Inches(3.1), Inches(3.1), Inches(2.0)], size=13,
      row_h=Inches(0.5), emphasis=lambda i, j, v: j == 3)
y2 = table_bottom(y + Inches(0.1), len(rows), row_h=Inches(0.5))
y2 = caption(s, y2, "Sign: + = K=0 higher, so negative = K=5 ahead. GRPO's K=5 advantage is "
                    "Holm-significant on both graders at iterations 4, 6, 7, 9 and 10 — largest "
                    "at the endpoint. PTO never significantly favours K=5 at any iteration on "
                    "either grader.")
y2 = band(s, y2 + Inches(0.12), "THE INTERACTION ITSELF",
          "Difference-in-differences at iteration 10: 0.718 (dz 0.793) on the primary, 0.815 "
          "(dz 0.972) on the held-out judge — both p_holm < .001. The training grader is not "
          "blind to it, and neither is the grader that was never the reward.",
          fill=VWASH, edge=VERDICT, label_color=VERDICT)
provenance(s, ["lookahead/reward/tables/k_table1.md", "lookahead/reward/tables/k_did.md"])


# ══════════════════════════════════════════════════════════════════════════════
# 9 · BEST FINAL STATE
# ══════════════════════════════════════════════════════════════════════════════
s, y = d.newslide("Exp3 · endpoint",
                  "GRPO K=5 @10 is the best final state on both graders")
rows = [
    ("GRPO  (K=5)", "4.517", "2.873", "0.210"),
    ("PTO  (K=5)", "4.307", "2.667", "0.264"),
    ("PTO  (K=0)", "4.260", "2.866", "0.491"),
    ("GRPO  (K=0)", "3.753", "2.257", "0.838"),
]
table(s, y + Inches(0.05),
      ["Final state @ 10", "Q1+Q2  primary", "Q1+Q2  held-out", "MICI ↓  primary"],
      rows, col_w=[Inches(3.0), Inches(2.6), Inches(2.6), Inches(2.6)], size=13,
      row_h=Inches(0.44), emphasis=lambda i, j, v: i == 0)
y2 = table_bottom(y + Inches(0.05), len(rows), row_h=Inches(0.44))
y2 = caption(s, y2, "On the primary, GRPO K=5 leads every one of the 8 instruments at the "
                    "endpoint and has the lowest MI-inconsistency of any final state.")
bandbot(s, "READ IT CAREFULLY",
        "“Still climbing” is a primary-grader statement: on the held-out judge the arm "
        "is flat since ~iteration 6 (I10 − I7 = −0.039, p = .65) and statistically tied with "
        "PTO K=0 at the top. And the measurement caveat, three slides on, travels with 4.517.",
        fill=CWASH, edge=CAVEAT, label_color=CAVEAT, size=12.5)
provenance(s, ["arms/outcomes/tables/gpt-4o-mini/leaderboard_scorecard.md",
               "arms/outcomes/tables/claude-haiku-4-5/leaderboard_scorecard.md"])


# ══════════════════════════════════════════════════════════════════════════════
# 10 · OVER-PRAISE, JUDGE-FREE (figure)
# ══════════════════════════════════════════════════════════════════════════════
figpage(d, "Exp3 · reward hacking",
        "The reward hack, measured without asking any judge",
        fig("lookahead", "behaviour", "figures", "overpraise_judgefree.png"),
        [("WHAT", "A deterministic lexical over-praise marker computed from the transcripts — "
                  "byte-identical under any grader — beside both graders' rated over-praise "
                  "rates."),
         ("AXIS", "Share of therapist turns containing at least one marker (not a per-turn "
                  "count)."),
         ("READ", "Both K=0 arms climb steeply in the late iterations; both K=5 arms stay near "
                  "zero."),
         ("N", "96 conversations per state.")],
        ["lookahead/behaviour/figures/overpraise_judgefree.png",
         "lookahead/behaviour/tables/overpraise_judgefree_data.md"])


# ══════════════════════════════════════════════════════════════════════════════
# 11 · HACK VERDICT
# ══════════════════════════════════════════════════════════════════════════════
s, y = d.newslide("Exp3 · reward hacking",
                  "Look-ahead suppresses over-praise in both optimizers")
rows = [
    ("GRPO  (K=0)", "0.671", "0.698"),
    ("PTO  (K=0)", "0.210", "0.299"),
    ("GRPO  (K=5)", "0.064", "0.051"),
    ("PTO  (K=5)", "0.045", "0.043"),
]
table(s, y + Inches(0.05),
      ["Arm @ 10", "lexical marker rate  (judge-free)", "rated over-praise rate  (primary)"],
      rows, col_w=[Inches(2.8), Inches(4.0), Inches(4.0)], size=13, row_h=Inches(0.42),
      emphasis=lambda i, j, v: i >= 2 and j > 0)
y2 = table_bottom(y + Inches(0.05), len(rows), row_h=Inches(0.42))
y2 = band(s, y2 + Inches(0.1), "VERDICT",
          "0.671 → 0.064 within GRPO (10.5x) and 0.210 → 0.045 within PTO (4.7x). The "
          "marker is computed from the transcripts, so no grader dispute applies — and PTO K=0, "
          "not GRPO K=5, is the second-most-sycophantic arm.",
          fill=VWASH, edge=VERDICT, label_color=VERDICT, size=14)
bandbot(s, "NOT THIS",
        "“K=5 stops the hack.” On the held-out grader, GRPO K=5's total "
        "MI-inconsistency still rises (0.326 → 0.628), at roughly half of K=0's rise; the "
        "primary reads the same conversations as flat. Say “slows the loop ~10x on "
        "over-praise”, per grader — never “stops”.",
        fill=CWASH, edge=CAVEAT, label_color=CAVEAT, size=12.5)
provenance(s, ["arms/validity/tables/gpt-4o-mini/overpraise_crosscheck.md",
               "arms/outcomes/tables/claude-haiku-4-5/leaderboard_scorecard.md"])


# ══════════════════════════════════════════════════════════════════════════════
# 12 · THE ASYMMETRY — the one-slide summary
# ══════════════════════════════════════════════════════════════════════════════
s, y = d.newslide("Exp3 · the one-slide summary", "What K=5 buys each optimizer")
panel_h = Inches(2.55)
half = (CW - Inches(0.3)) / 2
for i, (name, col, reward, hack) in enumerate([
    ("GRPO", GRPO_C,
     "Its best endpoint on both graders: +0.765 / +0.616 over K=0 (dz ~0.9 / 1.0), ahead on all "
     "8 instruments under the primary.",
     "Over-praise capture 10.5x lower (0.671 → 0.064, judge-free)."),
    ("PTO", PTO_C,
     "Nothing. No significant K=5 gain at any of 10 iterations on either grader; the endpoint "
     "contrast is null.",
     "Over-praise capture 4.7x lower (0.210 → 0.045, judge-free) — the lever still buys "
     "integrity where it buys no reward."),
]):
    x = ML + i * (half + Inches(0.3))
    rect(s, x, y, half, panel_h, WASH)
    rect(s, x, y, Inches(0.055), panel_h, col)
    tf = txbox(s, x + Inches(0.3), y + Inches(0.2), half - Inches(0.55), panel_h - Inches(0.4))
    run(para(tf, first=True), name, size=19, bold=True, color=col)
    p = para(tf, space_before=10)
    run(p, "Reward.  ", size=12.5, bold=True, color=INK)
    run(p, reward, size=12.5, color=BODY)
    p = para(tf, space_before=8)
    run(p, "Reward hacking.  ", size=12.5, bold=True, color=INK)
    run(p, hack, size=12.5, color=BODY)
y2 = y + panel_h + Inches(0.25)
band(s, y2, "SAME LEVER, DIFFERENT OPTIMIZER",
     "For GRPO, look-ahead is a performance lever. For PTO, it is only a hygiene lever. That "
     "asymmetry — not either method's win — is the finding of the experiment.",
     fill=VWASH, edge=VERDICT, label_color=VERDICT)
provenance(s, ["lookahead/reward/tables/k_table1.md",
               "arms/validity/tables/gpt-4o-mini/overpraise_crosscheck.md"])


# ══════════════════════════════════════════════════════════════════════════════
# 13 · MEASUREMENT HONESTY (figure)
# ══════════════════════════════════════════════════════════════════════════════
figpage(d, "Exp3 · measurement",
        "The caveat that travels with 4.517 — the training grader saturates on the winning arm",
        fig("measurement", "validity", "figures", "judge_saturation.png"),
        [("AGREEMENT", "Per-conversation cross-judge agreement on Q1 along GRPO K=5: 0.941 @I5 "
                       "→ 0.487 @I9 → 0.544 @I10 — the two lowest of all 44 states "
                       "(median 0.855)."),
         ("MECHANISM", "One-sided: the primary's Q1 SD collapses monotonically to 0.275x of base "
                       "(rho −0.86, p=.001); the held-out judge's SD on the SAME conversations "
                       "does not move."),
         ("SURVIVES", "The arm-level ranking: the held-out judge independently puts K=5 ahead at "
                      "dz 1.03. What breaks is the per-conversation ruler, exactly where the "
                      "primary scores highest."),
         ("RULE", "Never present the 4.517 endpoint without this slide.")],
        ["measurement/validity/figures/judge_saturation.png",
         "measurement/validity/tables/judge_saturation_data.md"])


# ══════════════════════════════════════════════════════════════════════════════
# 14 · REPLICATE DRAW
# ══════════════════════════════════════════════════════════════════════════════
s, y = d.newslide("Exp3 · robustness  (new since last meeting)",
                  "The replicate draw: every headline survives")
rows = [
    ("K lever @10   (GRPO K5 − K0)", "+0.765  →  +0.709 *", "+0.616  →  +0.637 *"),
    ("method @K0   (PTO − GRPO)", "+0.507  →  +0.516 *", "+0.609  →  +0.659 *"),
    ("method @K5   (PTO − GRPO)", "−0.210  →  −0.155 *", "−0.206  →  −0.227 *"),
    ("top pair   (GRPO K5 − PTO K0)", "+0.257  →  +0.193 *", "+0.007  →  −0.022   "
     "(n.s. both)"),
]
table(s, y + Inches(0.05),
      ["Contrast, Q1+Q2", "primary   orig → replicate", "held-out   orig → replicate"],
      rows, col_w=[Inches(4.3), Inches(3.6), Inches(3.6)], size=12.5, row_h=Inches(0.42))
y2 = table_bottom(y + Inches(0.05), len(rows), row_h=Inches(0.42))
y2 = caption(s, y2, "Replicate = a fresh 96-conversation generation from the same two adapters "
                    "(GRPO K5 @10, PTO K0 @10), scored on all 8 instruments by both graders. "
                    "* = Holm-significant in both draws. The K-lever and method rows re-draw one "
                    "side (the contested arm).")
bandbot(s, "AND A NOISE FLOOR, FOR THE FIRST TIME",
        "36 same-policy contrasts (2 arms x 9 metrics x 2 graders): 0 significant after Holm, "
        "max |dz| 0.216. The 4.517 headline reads 4.461 on the replicate; the held-out "
        "“tie” between GRPO K5 and PTO K0 is now demonstrated across two draws, not "
        "inferred from one gap.",
        fill=VWASH, edge=VERDICT, label_color=VERDICT, size=12.5)
provenance(s, ["measurement/replicate_draw.md"])


# ══════════════════════════════════════════════════════════════════════════════
# 15 · COMPUTE TRAJECTORY (figure)
# ══════════════════════════════════════════════════════════════════════════════
figpage(d, "Exp3 · compute axis",
        "The same curves on the axis money moves on — GPU-hours",
        fig("compute", "cost", "figures", "compute_trajectory.png"),
        [("X-AXIS", "Cumulative GPU-hours per arm, reconstructed from artifact mtimes (never "
                    "from the per-process metadata, which undercounts resumed iterations)."),
         ("COSTS", "PTO K0 8.119 h  ·  PTO K5 19.681 h  ·  GRPO K0 27.906 h  ·  GRPO K5 "
                   "51.205 h. Look-ahead costs 1.84x within GRPO; GRPO K5 costs 6.31x PTO K0."),
         ("READ", "An iteration is not a unit of spend — a whole PTO iteration costs a fraction "
                  "of a GRPO one, so matched-iteration and matched-budget answer different "
                  "questions.")],
        ["compute/cost/figures/compute_trajectory.png", "compute/cost/tables/compute_by_arm.md"])


# ══════════════════════════════════════════════════════════════════════════════
# 16 · COMPUTE VERDICT
# ══════════════════════════════════════════════════════════════════════════════
s, y = d.newslide("Exp3 · compute axis",
                  "At matched budget the verdict changes")
bullets(s, y + Inches(0.02), [
    ("GRPO's K at 51.2 GPU-h:", "K=5 > K=0 under all four grader select/evaluate combinations "
     "(Δ 0.256–0.435, every p_holm < .001), honest cross-grader selection included. The "
     "lever first draws level at ~23 GPU-h — below that it loses."),
    ("PTO's K at 19.7 GPU-h:", "K=5 < K=0 or no significant difference on all four combinations "
     "— the opposite sign, so the interaction holds at matched budget as well as matched "
     "iteration."),
    ("Method at PTO's budgets:", "PTO >> GRPO. At 8.1 GPU-h: 4/4 combinations (Δ "
     "0.76–0.90, dz 1.07–1.39). At 19.7: 3/4 — because at PTO's whole-run budget GRPO K=5 has "
     "only reached iteration 3."),
    ("The sharpest sentence:", "at 23.2 GPU-h GRPO K=5 is at iteration 4, scoring 4.120 / 2.784 "
     "— while PTO K=0's entire ten-iteration run costs 8.119 GPU-h and scores 4.260 / 2.866: "
     "higher on both graders at 35% of the compute."),
], size=12.5, gap=9)
bandbot(s, "BOTH AXES, ALWAYS",
        "At matched iteration GRPO K=5 wins; at any budget PTO can afford, PTO wins. Neither "
        "statement replaces the other — quote the axis with the claim.",
        fill=VWASH, edge=VERDICT, label_color=VERDICT, size=13)
provenance(s, ["compute/cost/tables/budget_sweep_crossjudge_verdicts.md",
               "compute/cost/tables/compute_by_arm.md"])


# ══════════════════════════════════════════════════════════════════════════════
# 17 · DIVIDER — ACT II
# ══════════════════════════════════════════════════════════════════════════════
d.divider("ACT II  ·  PAPERS",
          "Two drafts, disjoint by design",
          "P1 takes the narrowest decisive question first; P2 owns everything four-arm.",
          accent=GOLD)


# ══════════════════════════════════════════════════════════════════════════════
# 18 · P1 (figure page)
# ══════════════════════════════════════════════════════════════════════════════
figpage(d, "Paper 1 · ICLR 2027",
        "Scoring the Continuation — K-turn look-ahead rewards for GRPO",
        fig("lookahead", "reward", "figures", "k_headline_q1q2_grpo.png"),
        [("ARGUMENT", "Scoring a candidate turn by the K-turn continuation it leads to more than "
                      "doubles what group-relative RL extracts from the same oracle — and is the "
                      "difference between learning MI and learning to flatter the judge."),
         ("SCOPE", "The two GRPO arms ONLY. PTO appears nowhere — deliberately: every full-grid "
                   "statistic was recomputed on the 22 GRPO states, and the paper defers "
                   "optimizer generality to companion work."),
         ("STATUS", "Drafted end to end: 9-page ICLR body met, clean build, no TODOs, complete "
                    "NUMBERS.md ledger. Double-blind until camera-ready."),
         ("DEADLINES", "Abstract 18 Sep (3 weeks) · full paper 25 Sep 2026, both 23:59 UTC-12.")],
        ["papers/2026_grpo_lookahead_mi/", "lookahead/reward/figures/k_headline_q1q2_grpo.png"])


# ══════════════════════════════════════════════════════════════════════════════
# 19 · P1 MOVES + REFUSALS
# ══════════════════════════════════════════════════════════════════════════════
s, y = d.newslide("Paper 1 · the argument", "Five moves — and what the draft refuses to claim")
half = (CW - Inches(0.5)) / 2
tf = txbox(s, ML, y, half, Inches(0.36))
run(para(tf, first=True), "THE MOVES", size=11, bold=True, color=VERDICT)
bullets(s, y + Inches(0.38), [
    ("§4 Reward.", "K=5 beats K=0 on all 8 instruments under both graders; endpoint "
     "+0.765 / +0.616, gain vs base 2.3–2.6x K=0's."),
    ("§5 Cost, honest.", "1.92x per step, 1.835x per run; the lever loses below ~18 GPU-h "
     "and wins at the full budget under all four grader combinations."),
    ("§6 Behaviour.", "The 10.5x judge-free over-praise result. Stated per grader: "
     "“slows and roughly halves the loop”, never “stops”."),
    ("§7 Mechanism.", "Consistent-with, not shown: the K=5 training cut is a slightly "
     "better rank proxy; look-ahead rescales rather than sharpens the group signal."),
    ("§8 Measurement.", "The saturation caveat travels with the headline, in the paper "
     "itself."),
], size=11, gap=6, width=half, x=ML)
tf = txbox(s, ML + half + Inches(0.5), y, half, Inches(0.36))
run(para(tf, first=True), "THE REFUSALS", size=11, bold=True, color=CAVEAT)
bullets(s, y + Inches(0.38), [
    "Not a claim about optimizers in general — PTO is never named; transfer is deferred to the "
    "companion paper.",
    "Not “eliminates reward hacking” — the per-grader split is stated.",
    "No dose–response: K in {0, 5} by design, and the limitations say so.",
    "No clinical claim — every instrument is LLM-administered; no human MI coder has rated any "
    "conversation.",
    "No patient-generalisation claim — all 96 personas are in-sample.",
], size=11, gap=6, width=half, x=ML + half + Inches(0.5))
provenance(s, ["papers/2026_grpo_lookahead_mi/README.md",
               "papers/2026_grpo_lookahead_mi/NUMBERS.md"])


# ══════════════════════════════════════════════════════════════════════════════
# 20 · P2
# ══════════════════════════════════════════════════════════════════════════════
s, y = d.newslide("Paper 2 · ARR October",
                  "Same Lever, Different Optimizer — the full 2x2")
bullets(s, y + Inches(0.02), [
    ("The interaction.", "The optimizer ranking flips with the reward horizon — DiD dz 0.79 / "
     "0.97, both graders, every cell surviving the replicate draw."),
    ("The budget axis.", "PTO K=0 matches GRPO K=5's held-out gain at 6.31x less compute; the "
     "reversals between matched-iteration and matched-budget are part of the claim."),
    ("The behaviour story.", "Both K=0 arms flatter; look-ahead suppresses over-praise in both "
     "optimizers — this is where PTO's “hygiene lever” result lives, the arm of the "
     "story P1 deliberately leaves out."),
    ("Also inside.", "The ICLR-era regime re-scored under the modern grader, and the full-grid "
     "measurement section (sign preservation 88.4% over 7,568 contrasts)."),
    ("Status.", "Drafted 2026-08-26 in ACL long format, 8-page body met, fresh NUMBERS.md. "
     "Target: the ARR October 2026 cycle, feeding NAACL 2027 + COLING 2027."),
], size=12.5, gap=9)
bandbot(s, "DISJOINT SCOPES",
        "P1 never names PTO; P2 owns everything four-arm. Shared numbers are cited from the same "
        "EDA artifacts in both ledgers, and neither paper imports the other's argument.",
        fill=VWASH, edge=VERDICT, label_color=VERDICT, size=12.5)
provenance(s, ["papers/2026_pto_grpo_mi/README.md", "papers/README.md"])


# ══════════════════════════════════════════════════════════════════════════════
# 21 · DIVIDER — ACT III
# ══════════════════════════════════════════════════════════════════════════════
d.divider("ACT III  ·  EXP4",
          "The cost ceiling, removed",
          "The same science on a fully open model stack — an arm costs $0 in API.",
          accent=MINT)


# ══════════════════════════════════════════════════════════════════════════════
# 22 · EXP4 — WHY
# ══════════════════════════════════════════════════════════════════════════════
s, y = d.newslide("Exp4 · why it exists",
                  "Exp3's binding constraint was the API bill")
bullets(s, y + Inches(0.02), [
    ("What happened.", "The OpenAI balance hit zero on 24 Aug, killing GRPO K=5's iteration 10 "
     "at step 70/136; the 20 Aug spend cap had already killed two sessions. Cost — not GPU — "
     "capped every design decision in Exp3 (iterations, G/M, the held RQ-iii)."),
    ("The move.", "Same science — same 1B therapist, same 96 personas and rubrics verbatim — "
     "but patient, oracle and judge are an open Gemma-4 model (E4B default) behind one local "
     "vLLM server. A full arm costs $0 in API; only Colab GPU-hours."),
    ("The guard.", "The grader IS the instrument, so an oracle-sanity gate blocks any arm until "
     "the Gemma grader proves non-degenerate spread and rank agreement against frozen "
     "gpt-4o-mini reference scores. An open grader can honour the schema and still measure "
     "nothing — that failure is silent, hence the gate."),
    ("Status.", "Code complete and locally gated: 94 smoke checks + the EDA self-check green, "
     "real DPO and GRPO steps run on the local card. Nothing trained yet; no data exists."),
    ("By construction.", "Five Exp3 defects fixed from day 1: per-attempt API timeouts, stable "
     "persona-id filenames, append-only per-phase timing, a per-state parquet score lake, "
     "append-only run-metadata history."),
], size=12, gap=8)
bandbot(s, "SCOPE",
        "A side project — not a thesis chapter unless the results earn it. Different grader = "
        "different score axis: Exp4 numbers are never comparable to Exp3's.",
        fill=CWASH, edge=CAVEAT, label_color=CAVEAT, size=12.5)
provenance(s, ["Exp4_OpenStack/CLAUDE.md"])


# ══════════════════════════════════════════════════════════════════════════════
# 23 · EXP4 — WHAT IT UNLOCKS
# ══════════════════════════════════════════════════════════════════════════════
s, y = d.newslide("Exp4 · what it unlocks", "The experiments Exp3 could not afford")
bullets(s, y + Inches(0.02), [
    ("RQ-iii, finally.", "The oracle-instrument sweep — train on WAI-SR, CSQ-8 or MITI instead "
     "of Q1+Q2. Held in Exp3 purely for cost; the training questionnaire is a first-class knob "
     "in Exp4's arm grammar."),
    ("K as a dose.", "K in {1, 2, 3, ...} between the two endpoints — the dose–response curve "
     "P1 explicitly refuses to claim on two points."),
    ("Training-run replicates.", "Exp3 has ONE training run per arm; the replicate draw bounds "
     "evaluation noise only. At $0 API, seed replicates of whole arms become affordable."),
    ("Longer runs.", "GRPO K=5 was still climbing on the primary at iteration 10 — the "
     "truncation was budget, not convergence."),
    ("The gate ladder.", "Colab roles smoke → full oracle-sanity on both Gemma sizes "
     "(choose the grader by rank agreement + spread) → 2-iteration mini-arm → first "
     "real arm (GRPO K=0)."),
], size=12.5, gap=9)
bandbot(s, "THE ASK",
        "One Colab session to run the gate ladder end to end — GPU-hours only, $0 API. If the "
        "Gemma grader passes sanity, the first real arm can start immediately after.",
        fill=VWASH, edge=VERDICT, label_color=VERDICT, size=13)
provenance(s, ["Exp4_OpenStack/CLAUDE.md"])


# ══════════════════════════════════════════════════════════════════════════════
# 24 · DECISIONS & TIMELINE
# ══════════════════════════════════════════════════════════════════════════════
s, y = d.newslide("Decisions", "Three decisions, one calendar")
rows = [
    ("1", "P1 — Scoring the Continuation", "ICLR 2027: abstract 18 Sep (3 weeks), paper 25 Sep",
     "Freeze scope now; co-author list + final read-through; remaining items are its README's "
     "open list."),
    ("2", "P2 — Same Lever, Different Optimizer", "ARR October 2026 cycle",
     "Polish after P1 freezes; the drafts share numbers only through the EDA artifacts."),
    ("3", "Exp4 gate ladder", "one Colab session, $0 API",
     "Go / no-go: run now in the background of writing, or hold until after the P1 deadline."),
]
table(s, y + Inches(0.05), ["", "What", "When", "The decision"],
      rows, col_w=[Inches(0.5), Inches(3.6), Inches(3.3), Inches(4.1)], size=11.5,
      row_h=Inches(0.72), prose_cols=(1, 2, 3), emphasis=lambda i, j, v: j == 1)
y2 = table_bottom(y + Inches(0.05), len(rows), row_h=Inches(0.72))
y2 = band(s, y2 + Inches(0.15), "SUGGESTED ORDER",
          "P1 first — it is the nearest deadline and needs no new data. The Exp4 gates are a "
          "single session and can run while writing; any real Exp4 arm waits until P1 is "
          "submitted.",
          fill=VWASH, edge=VERDICT, label_color=VERDICT)
provenance(s, ["STATUS.md § Next step", "papers/README.md"])


# ══════════════════════════════════════════════════════════════════════════════
# 25 · CLOSING
# ══════════════════════════════════════════════════════════════════════════════
s = d._next()
rect(s, 0, 0, W, H, INK)
rect(s, ML, Inches(1.9), Inches(1.5), Inches(0.05), GRPO_C)
tf = txbox(s, ML, Inches(2.25), CW - Inches(0.8), Inches(3.6))
p = para(tf, first=True)
run(p, "Look-ahead is a performance lever for GRPO", size=28, bold=True, color=PAPER)
p = para(tf, space_before=2)
run(p, "and a hygiene lever for PTO.", size=28, bold=True, color=SKY)
p = para(tf, space_before=16)
run(p, "Which optimizer wins depends on which K — and which axis — you ask.", size=17,
    color=MIST)
p = para(tf, space_before=8)
run(p, "The grid that settles this is complete, replicated at its contested corners, and "
       "priced.", size=17, color=MIST)
p = para(tf, space_before=22)
run(p, "Next: P1 to ICLR 2027 (abstract in 3 weeks)  ·  P2 to ARR October  ·  the Exp4 gate "
       "ladder, one Colab session.", size=13, color=SLATE)

d.save(OUT, repo=REPO)
