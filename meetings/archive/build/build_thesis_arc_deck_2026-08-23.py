#!/usr/bin/env python
"""
build_thesis_arc_deck_2026-08-23.py — the 2026-08-23 supervisor deck.

THE THREE-EXPERIMENT DECK. Every previous builder here presents Exp3 and treats Exp1 as a one-slide
reminder and Exp2 as a footnote about quantization. This one is organised as the thesis is: three
acts, one per experiment, each closing on *what it established* and *what it could not settle* —
which is what motivates the next.

    ACT I    Exp1 (ICLR 2025)  — PTO, published. Re-audited against its own shipped data and
                                 re-scored under the modern grader.
    ACT II   Exp2              — PTO across three training oracles. Largely a negative result.
    ACT III  Exp3              — PTO vs GRPO, both K, both graders, both cost axes.

⚠ **Exp1 and Exp2 are PTO-ONLY.** Exp2 also contains a GRPO V1 run; **it had a bug and its results
are void** (root CLAUDE.md § Methods, Exp2_PTO/CLAUDE.md § "GRPO V1 — VOID"). It is absent from this
deck deliberately — do not "helpfully" restore it in a later revision. The PTO-vs-GRPO comparison is
an Exp3 result and only an Exp3 result, because only there are the two methods iterative, sharing
`code/_shared/`, at matched MCL / K / M = G = 8 / temperature / oracle.

NUMBERS ARE HARD-CODED, like every builder here: a deck is a snapshot of what was presented on its
date, not a live view. Every figure below was gathered by a cold read of the owning table or a
direct recomputation from raw data, then independently re-verified against the cited source. Where a
number contradicts what a project doc asserts, the TABLE wins and the slide says so.

⚠ Two live EDA defects are ROUTED AROUND, not quoted (STATUS.md § "Where the artifacts live"):
  - the hardcoded `CENSOR_NOTE` in eight modules still says GRPO_LA5 is censored at iteration 5, and
    33 rendered CAPTIONS.md files repeat it. The DATA is right (iteration 6 exists and is scored);
    the prose beside it is stale. No caption text is reproduced on any slide.
  - `faithfulness.py:110`'s asymmetric SERIES pools iterations 1-6 for GRPO_LA5 under a column
    labelled 1-5. Nothing from `lookahead/mechanism/tables/faithfulness_curve*.md` is used here.

Owning artifacts, under ``Exp3_PTO_GRPO/eda/results/`` unless marked otherwise:

    ACT I   lookahead/replication/tables/crossgen_{levels,kcontrast,kcontrast_summary,vsbase}.md
            lookahead/replication/tables/{sd_summary,sd_tally,ceiling}.md
            Exp1_ICLR2025/paper.pdf (Tables 1, 2, 4)  ·  Exp1_ICLR2025/data/**/scores_*.csv
    ACT II  meetings/build/_exp2_summary.{md,csv}  ← computed from Exp2_PTO/eda/eval/**
    ACT III arms/outcomes/tables/<judge>/leaderboard_scorecard.md
            lookahead/reward/tables/{k_table1,k_did,k_endpoints}.md
            lookahead/transfer/tables/k_retention_summary.md
            method/contrast/tables/method_paired_{by_K,best}.md
            compute/cost/tables/compute_by_{arm,iteration}.md
            measurement/validity/tables/multijudge_{sign_preservation,variance_components}.md

Build:
    & ..\\..\\.venv\\Scripts\\python.exe build_thesis_arc_deck_2026-08-23.py
    .\\export_pdf.ps1 ..\\2026-08-23\\thesis_arc_2026-08-23.pptx
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from _deck_kit import (  # noqa: E402
    BODY, CAVEAT, CW, CWASH, Deck, DUSK, GOLD, GRPO_C, H, INK, MINT, MIST, ML, MUTED, PANEL,
    PAPER, PTO_C, SKY, SLATE, VERDICT, VWASH, W, WASH,
    band, bandbot, bullets, caption, figband, para, pic, provenance, rect, run, table,
    table_bottom, txbox,
)
from pptx.enum.text import PP_ALIGN  # noqa: E402
from pptx.util import Inches  # noqa: E402

REPO = os.path.dirname(os.path.dirname(HERE))
ROOT = os.path.join(REPO, "Exp3_PTO_GRPO")
RES = os.path.join(ROOT, "eda", "results")
OUTDIR = os.path.join(REPO, "meetings", "2026-08-23")
OUT = os.path.join(OUTDIR, "thesis_arc_2026-08-23.pptx")

EXP1_C = GOLD     # amber-gold — Exp1
EXP2_C = PTO_C    # teal — Exp2 is PTO, same identity PTO carries in Act III


def _rel(path):
    return os.path.relpath(path, RES).replace(os.sep, "/")


def fig(*parts):
    return os.path.join(RES, *parts)


d = Deck()


# ══════════════════════════════════════════════════════════════════════════════
# 1 · TITLE
# ══════════════════════════════════════════════════════════════════════════════
s = d._next()
rect(s, 0, 0, W, H, INK)
rect(s, ML, Inches(1.72), Inches(1.5), Inches(0.05), EXP1_C)
tf = txbox(s, ML, Inches(2.05), CW - Inches(1.2), Inches(2.6))
p = para(tf, first=True)
run(p, "Three experiments,", size=44, bold=True, color=PAPER)
p = para(tf, space_before=2)
run(p, "and what each one actually settled", size=44, bold=True, color=SKY)
p = para(tf, space_before=20)
run(p, "Look-ahead depth  ·  optimizer family  ·  the oracle instrument", size=16, color=MIST)
tf = txbox(s, ML, H - Inches(1.66), CW, Inches(1.1))
p = para(tf, first=True)
run(p, "Lior Baruch   ·   Reichman University   ·   23 August 2026", size=13, color=SLATE)
p = para(tf, space_before=6)
run(p, "Exp1  Llama-2-7B / GPT-3.5      Exp2  Llama-3.2-1B 4-bit / gpt-4o-mini      "
       "Exp3  Llama-3.2-1B bf16 / gpt-4o-mini, Claude Haiku 4.5 held out",
    size=10.5, color=DUSK)
p = para(tf, space_before=5)
run(p, "Exp1 and Exp2 are PTO only. GRPO enters in Exp3.", size=10.5, italic=True, color=DUSK)


# ══════════════════════════════════════════════════════════════════════════════
# 2 · THE ARC
# ══════════════════════════════════════════════════════════════════════════════
s, y = d.newslide("Agenda", "Three experiments, and the question each one handed forward")
y += Inches(0.02)
for num, name, col, got, left in [
    ("I", "Exp1 — ICLR 2025", EXP1_C,
     "PTO beats the untrained baseline, and deeper look-ahead looks better.",
     "One model was patient, oracle AND training reward. Best-vs-best was chosen post hoc. "
     "The depth contrast is significant on one of four metrics."),
    ("II", "Exp2", EXP2_C,
     "Harder patients, a JSON-schema oracle, three training instruments.",
     "Only the Q1+Q2-trained arm moved. Training on WAI-SR or CSQ-8 did not improve WAI-SR or "
     "CSQ-8. Look-ahead was a wash at matched iterations."),
    ("III", "Exp3", GRPO_C,
     "Look-ahead helps GRPO and hurts PTO; which optimizer wins depends on K — and both answers "
     "hold on a grader that was never the reward.",
     "One arm stopped at iteration 6. Every contested endpoint is a single 96-conversation draw. "
     "Which instrument to TRAIN on is still untested here."),
]:
    hgt = Inches(1.50)
    rect(s, ML, y, CW, hgt, WASH)
    rect(s, ML, y, Inches(0.055), hgt, col)
    tfn = txbox(s, ML + Inches(0.32), y + Inches(0.26), Inches(1.0), Inches(0.8))
    run(para(tfn, first=True), num, size=25, bold=True, color=col)
    tfq = txbox(s, ML + Inches(1.28), y + Inches(0.15), CW - Inches(1.75), Inches(1.3))
    run(para(tfq, first=True), name, size=16.5, bold=True, color=INK)
    pg = para(tfq, space_before=5)
    run(pg, "Established.  ", size=11.5, bold=True, color=VERDICT)
    run(pg, got, size=11.5, color=INK)
    pl = para(tfq, space_before=4)
    run(pl, "Left open.  ", size=11.5, bold=True, color=CAVEAT)
    run(pl, left, size=11.5, color=BODY)
    y += hgt + Inches(0.16)
tf = txbox(s, ML, y + Inches(0.02), CW, Inches(0.5))
run(para(tf, first=True),
    "Every number that follows was read off a table or recomputed from raw data, then verified a "
    "second time against that source. Where one disagrees with a project doc, the slide says so.",
    size=11.5, italic=True, color=MUTED)


# ══════════════════════════════════════════════════════════════════════════════
# 3 · WHAT IS CONSTANT ACROSS ALL THREE
# ══════════════════════════════════════════════════════════════════════════════
s, y = d.newslide("Setup", "What never changed across the three experiments")
left_w = CW * 0.53
bullets(s, y, [
    ("The task.", "A therapist LLM runs a Motivational Interviewing session against a simulated "
                  "patient. No human is in the loop at any point."),
    ("The personas.", "96 patients, factorial: gender × cooperation × problem × duration × prior "
                      "attempts × age = 2·3·2·2·2·2. Identical construction in all three "
                      "experiments, so every contrast is paired on persona, n = 96."),
    ("The reward.", "A larger LLM grades the finished transcript on validated MI questionnaires. "
                    "Q1 = session satisfaction (5 items); Q2 = working alliance (17 items)."),
    ("The lever.", "K = look-ahead depth. Before scoring a candidate reply, simulate K more turns "
                   "and score where the conversation ENDS UP, not how the reply looks alone."),
], size=13, gap=11, width=left_w)

bx = ML + left_w + Inches(0.4)
bw = CW - left_w - Inches(0.4)
rect(s, bx, y, bw, Inches(3.85), WASH)
tfb = txbox(s, bx + Inches(0.3), y + Inches(0.24), bw - Inches(0.6), Inches(3.4))
run(para(tfb, first=True), "WHAT CHANGED, AND WHY", size=9.5, bold=True, color=MUTED)
for lead, rest in [
    ("Exp1 → Exp2", "one model played patient, oracle and reward. Split the patient off, "
                    "harden it, and constrain the oracle to a JSON schema."),
    ("Exp1 → Exp2", "7B → 1B therapist. Can a small model be taught this at all?"),
    ("Exp2 → Exp3", "add a SECOND grader that never played the patient and was never the reward."),
    ("Exp2 → Exp3", "add GRPO, so PTO has something to be measured against."),
]:
    pp = para(tfb, space_before=12)
    run(pp, lead, size=11.5, bold=True, color=INK)
    run(pp, "   " + rest, size=11, color=BODY)
pn = para(tfb, space_before=15)
run(pn, "Each experiment is a fresh re-implementation. No data flows between them.",
    size=10.5, italic=True, color=CAVEAT)


# ══════════════════════════════════════════════════════════════════════════════
# ACT I — Exp1
# ══════════════════════════════════════════════════════════════════════════════
d.divider("ACT I", "Exp1 — the published result",
          "Llama-2-7B therapist, GPT-3.5 as patient, oracle and reward. PTO at K ∈ {0, 5}, "
          "7 iterations each, 96 personas. Published at ICLR 2025.", accent=EXP1_C)

# ── 5 · what was published ────────────────────────────────────────────────────
s, y = d.newslide("Act I · Exp1", "What the paper reports")
y = band(s, y, "THE PUBLISHED CLAIM",
         "Every PTO model beats the untrained baseline, and the K=5 models sit above the K=0 "
         "models on every metric in Table 1.",
         fill=VWASH, edge=VERDICT, label_color=VERDICT)
rows = [
    ["Base  (untrained Llama-2-7B)", "3.521", "3.385", "3.453", "0.740", "43.7"],
    ["K=0, best iteration (M4)", "3.969", "3.585", "3.777", "0.769", "38.3"],
    ["K=5, best iteration (M7)", "4.190", "3.775", "3.982", "0.414", "34.4"],
]
table(s, y + Inches(0.08), ["model", "Q1", "Q2", "Final", "Final SD", "turns"], rows,
      col_w=[Inches(3.5), Inches(1.35), Inches(1.35), Inches(1.35), Inches(1.5), Inches(1.35)],
      emphasis=lambda i, j, v: i == 2)
ty = table_bottom(y + Inches(0.08), 3)
ty = caption(s, ty, "Q1 = session satisfaction (5 items), Q2 = working alliance (17 items), "
                    "both 1–5.  Final = mean(Q1, Q2).  n = 96 per row.  "
                    "Turn counts are from the prose, not a table — the paper publishes no "
                    "length table.")
bullets(s, ty + Inches(0.04), [
    ("All 15 rows reproduce exactly from the conversation data still in the repo.",
     "90 statistics, n = 96 each, matching to the third decimal (max deviation 0.0005). "
     "The measurement layer of Exp1 is sound — everything questioned on the next slide is in the "
     "inference layer."),
], size=12.5, gap=8)
provenance(s, ["Exp1_ICLR2025/paper.pdf p.8 Table 1, p.9 §5.2",
               "lookahead/replication/tables/crossgen_levels.md"])

# ── 6 · what the paper's own statistics support ───────────────────────────────
s, y = d.newslide("Act I · Exp1", "What the paper's own statistics actually support")
y = band(s, y, "THE NARROWER TRUE CLAIM",
         "PTO beats the baseline — that is solid and significant. But K=5 over K=0 is significant "
         "on ONE metric of four: working alliance, p = 0.0315.",
         fill=CWASH, edge=CAVEAT, label_color=CAVEAT)
rows = [
    ["Final Score", "+0.324  **", "+0.529  ***", "+0.206", ".0807", "no"],
    ["Q1 · satisfaction", "+0.448  **", "+0.669  ***", "+0.221", ".2095", "no"],
    ["Q2 · working alliance", "+0.199  *", "+0.390  ***", "+0.191", ".0315", "YES"],
    ["Conversation length", "−5.34  *", "−9.28  ***", "−3.94", ".0992", "no"],
]
table(s, y + Inches(0.06),
      ["metric", "Base → K=0", "Base → K=5", "K=0 → K=5", "p", "significant?"],
      rows, col_w=[Inches(2.7), Inches(1.75), Inches(1.75), Inches(1.75), Inches(1.35),
                   Inches(1.65)],
      emphasis=lambda i, j, v: i == 2 and j in (3, 4, 5))
ty = table_bottom(y + Inches(0.06), 4)
ty = caption(s, ty, "Tukey HSD, paper Table 4.  * p<.05  ** p<.01  *** p<.001.  "
                    "The K=0 → K=5 column compares the two arms' best iterations, chosen after "
                    "seeing the results.")
bullets(s, ty + Inches(0.02), [
    ("The omnibus ANOVA was run on 3 groups, not 15.", "Recomputing on Base + L0_M4 + L5_M7 "
     "reproduces all four published F-statistics exactly (F = 15.637 on Final). The honest "
     "all-15-model ANOVA gives F = 4.299 — still highly significant (p = 1.6e-07), so the "
     "conclusion survives; the presentation does not."),
], size=12.5, gap=8)
provenance(s, ["Exp1_ICLR2025/paper.pdf p.9 Table 2, p.13 Table 4",
               "recomputed from Exp1_ICLR2025/data/conversations_eval/**/scores_*.csv"])

# ── 7 · re-scored under the modern grader ─────────────────────────────────────
s, y = d.newslide("Act I · Exp1", "The same transcripts, re-scored four years later")
y = band(s, y, "IT REPLICATES — IN DIRECTION",
         "Re-scoring all 1,440 Exp1 conversations with gpt-4o-mini keeps K=5 above K=0 at 7 of 7 "
         "iterations. The ordering is a property of the regime, not of GPT-3.5.",
         fill=VWASH, edge=VERDICT, label_color=VERDICT)
rows = [
    ["Averaged over all 7 iterations", "−0.132", "−0.543", "<.001", "−0.206", "−0.612", "<.001"],
    ["The paper's best-vs-best pair", "−0.129", "−0.250", ".006", "−0.206", "−0.251", ".331"],
    ["Each arm at its own best", "−0.129", "−0.191", ".405", "−0.206", "−0.251", ".331"],
]
table(s, y + Inches(0.08),
      ["contrast (K=0 − K=5)", "Δ modern", "dz", "p", "Δ original", "dz", "p"],
      rows, col_w=[Inches(3.5), Inches(1.4), Inches(1.05), Inches(1.1), Inches(1.4),
                   Inches(1.05), Inches(1.1)],
      emphasis=lambda i, j, v: i == 0)
ty = table_bottom(y + Inches(0.08), 3)
ty = caption(s, ty, "Metric = Final (mean of Q1, Q2).  Negative = K=5 higher.  Persona-paired, "
                    "n = 96.  'modern' = the gpt-4o-mini re-scoring; 'original' = the GPT-3.5 "
                    "scores Exp1 saved beside each conversation — not the thesis's held-out "
                    "judge, which never saw Exp1.  p = Wilcoxon signed-rank.")
bandbot(s, "NOT THIS",
        "Not that any single K=5 checkpoint beats any single K=0 checkpoint. Only the ARM-level "
        "average survives both graders with a CI clear of zero — and the paper's claim that every "
        "K=5 model outscores every K=0 model is false under BOTH graders (the first K=5 iteration "
        "breaks it).",
        fill=CWASH, edge=CAVEAT, label_color=CAVEAT, size=12.5)
provenance(s, ["lookahead/replication/tables/crossgen_kcontrast_summary.md"])

# ── 8 · the grader was doing more work than anyone thought ────────────────────
s, y = d.newslide("Act I · Exp1", "Swapping the grader changes which arm has a result")
figband(s, y, fig("lookahead", "replication", "figures", "crossgen.png"), "READ IT AS",
        "Under GPT-3.5, 8 of 14 trained models separate from base after correction — including the "
        "K=0 arm's iteration 4, the paper's K=0 champion. Under gpt-4o-mini only 4 do, and ALL "
        "FOUR are K=5. The modern grader erases the K=0 arm's only significant gain.",
        fill=WASH, edge=EXP1_C, label_color=MUTED, size=12.5, bold_text=False)
provenance(s, [_rel(fig("lookahead", "replication", "figures", "crossgen.png")),
               "lookahead/replication/tables/crossgen_vsbase.md"])

# ── 9 · the stability claim ───────────────────────────────────────────────────
s, y = d.newslide("Act I · Exp1", "The 'more stable' claim is a scoring ceiling")
y = band(s, y, "WHAT THE PAPER SAID",
         "The best K=5 model has the lowest standard deviation on every metric — read as evidence "
         "that look-ahead produces more consistent therapists.",
         fill=CWASH, edge=CAVEAT, label_color=CAVEAT, size=14)
bullets(s, y + Inches(0.04), [
    ("On Exp1's own numbers the ordering is real but not clean.", "Lowest SD is indeed the K=5 "
     "endpoint (0.408). But the widest K=5 model (0.550) is wider than the tightest K=0 model "
     "(0.523) — the same first-iteration model that breaks the mean ordering. No variance test "
     "was run anywhere in the paper."),
    ("Re-tested properly on Exp3's 36 trained states, SD is predicted by the MEAN.",
     "Spearman(mean, SD) = −0.863, p = 1.4e-11 under the training oracle. Higher-scoring models "
     "are tighter because they are pressed against the top of a 1–5 scale."),
    ("The held-out judge, which has no ceiling, barely shows the effect.", "Same correlation is "
     "only −0.376. That gap is the evidence: Claude Haiku never awards ≥ 4.5 on Q1+Q2 in any of "
     "the 40 Exp3 model states, so it has nothing to compress against."),
    ("Concretely, at the best PTO state:", "the cooperative third of personas scores 4.904 with "
     "SD 0.048 and 100% of conversations above 4.5. That is saturation, not stability. The "
     "resistant third of the same model still has SD 0.365."),
], size=12.5, gap=10)
provenance(s, ["lookahead/replication/tables/{sd_summary,sd_tally,ceiling}.md",
               "crossgen_levels.md"])

# ── 10 · Act I close ──────────────────────────────────────────────────────────
s, y = d.newslide("Act I · Exp1", "What Exp1 settled, and what it handed forward")
half = CW / 2 - Inches(0.16)
rect(s, ML, y, half, Inches(4.30), VWASH)
rect(s, ML, y, Inches(0.055), Inches(4.30), VERDICT)
tfa = txbox(s, ML + Inches(0.3), y + Inches(0.22), half - Inches(0.6), Inches(3.9))
run(para(tfa, first=True), "SETTLED", size=9.5, bold=True, color=VERDICT)
for t in ["PTO trains a therapist LLM. Both arms beat the untrained baseline on every metric, "
          "with large and significant effects.",
          "Look-ahead points the right way. K=5 sits above K=0 at 7 of 7 iterations, and it "
          "survives a grader swap four years later.",
          "Trained models hold shorter sessions — 43.7 turns down to 34.4.",
          "The data is trustworthy. Every published cell reproduces from the shipped files."]:
    pp = para(tfa, space_before=13)
    run(pp, t, size=12.5, color=INK)

rect(s, ML + half + Inches(0.32), y, half, Inches(4.30), CWASH)
rect(s, ML + half + Inches(0.32), y, Inches(0.055), Inches(4.30), CAVEAT)
tfb = txbox(s, ML + half + Inches(0.62), y + Inches(0.22), half - Inches(0.6), Inches(3.9))
run(para(tfb, first=True), "HANDED FORWARD", size=9.5, bold=True, color=CAVEAT)
for t in ["One model was patient, oracle and reward at once. There is no measurement independent "
          "of the thing being optimised.",
          "K=5 over K=0 is significant on one metric of four, between two iterations picked after "
          "seeing the results.",
          "The scale ceilings out. On cooperative personas the oracle has nowhere left to score.",
          "It cannot be reproduced from its own text. The branching factor, trees per iteration "
          "and max length appear in the algorithms as symbols and are never given values; no DPO "
          "beta, learning rate, LoRA config, temperature or seed is stated anywhere.",
          "A 7B therapist against cooperative patients is the easy regime. Does any of it hold "
          "when the model is smaller and the patients push back?"]:
    pp = para(tfb, space_before=13)
    run(pp, t, size=12.5, color=INK)


# ══════════════════════════════════════════════════════════════════════════════
# ACT II — Exp2
# ══════════════════════════════════════════════════════════════════════════════
d.divider("ACT II", "Exp2 — the regime change",
          "Llama-3.2-1B therapist in 4-bit. gpt-4o-mini as patient and oracle, on a JSON schema. "
          "Less cooperative patients. PTO across three training instruments × K ∈ {0, 5}.",
          accent=EXP2_C)

# ── 12 · what changed ─────────────────────────────────────────────────────────
s, y = d.newslide("Act II · Exp2", "Four things changed at once")
rows = [
    ["Therapist", "Llama-2-7B", "Llama-3.2-1B, 4-bit NF4", "can a 1B model learn this?"],
    ["Patient", "GPT-3.5, cooperative", "gpt-4o-mini, resistant", "remove the easy regime"],
    ["Oracle", "GPT-3.5, regex-parsed", "gpt-4o-mini, JSON schema", "stop losing scores to parsing"],
    ["Instruments", "Q1 + Q2", "6 scored, 3 trained on", "does the rubric choice matter?"],
]
table(s, y + Inches(0.04), ["", "Exp1", "Exp2", "the question it opens"], rows,
      col_w=[Inches(1.7), Inches(2.6), Inches(3.0), Inches(4.0)],
      prose_cols=(1, 2, 3), row_h=Inches(0.46))
ty = table_bottom(y + Inches(0.04), 4, row_h=Inches(0.46))
bullets(s, ty + Inches(0.06), [
    ("Because four things changed together, Exp2 cannot attribute anything.", "It is a regime "
     "change, not a controlled comparison — which is exactly why Exp3 holds everything fixed and "
     "moves one lever at a time."),
    ("Scale of the sweep:", "50 model states × 96 conversations × 6 questionnaires = 28,800 score "
     "files on disk, covering 4,800 scored conversations."),
], size=12.5, gap=9)
bandbot(s, "CORRECTION",
        "Project docs describe Exp2 as “4 oracles × K” over “4,512 convs / 47 models”. Neither "
        "reproduces: only THREE training instruments exist on disk (Q1+Q2, WAI-SR, CSQ-8), and the "
        "census is 4,800 conversations across 50 model states.",
        fill=CWASH, edge=CAVEAT, label_color=CAVEAT, size=12.5)
provenance(s, ["meetings/build/_exp2_summary.md §1, §8  ←  Exp2_PTO/eda/eval/**"])

# ── 13 · the sweep ────────────────────────────────────────────────────────────
s, y = d.newslide("Act II · Exp2", "PTO still works — but only when trained on Q1+Q2")
y = band(s, y, "THE RESULT",
         "Every arm improves on the Q1+Q2 axis. Only the arm actually TRAINED on Q1+Q2 improves "
         "convincingly; the WAI-SR arms do not clear p < .05 at all.",
         fill=VWASH, edge=VERDICT, label_color=VERDICT)
rows = [
    ["Q1+Q2", "K=0", "2.771", "+0.393", ".0006", "yes"],
    ["Q1+Q2", "K=5", "2.968", "+0.590", "<.0001", "yes"],
    ["WAI-SR", "K=0", "2.614", "+0.237", ".075", "no"],
    ["WAI-SR", "K=5", "2.590", "+0.212", ".096", "no"],
    ["CSQ-8", "K=0", "2.629", "+0.251", ".022", "yes"],
    ["CSQ-8", "K=5", "2.599", "+0.221", ".014", "yes"],
]
table(s, y + Inches(0.06),
      ["trained on", "K", "best Q1+Q2", "vs base", "p", "clears .05?"],
      rows, col_w=[Inches(2.1), Inches(1.0), Inches(1.85), Inches(1.55), Inches(1.35),
                   Inches(1.65)],
      emphasis=lambda i, j, v: i in (0, 1))
ty = table_bottom(y + Inches(0.06), 6)
caption(s, ty, "Base = 2.378.  Each arm's best iteration, persona-paired against base, n = 96, "
               "Wilcoxon.  The K=5 Q1+Q2 arm ran to iteration 10; every other arm stops at 5.")
bandbot(s, "NOT THIS",
        "Not that K=5 beats K=0 here. The 2.968 arm had five more training iterations than its "
        "K=0 sibling. Matched iteration for iteration, K is a wash: mean Δ +0.060 (Q1+Q2), "
        "−0.020 (WAI-SR), +0.018 (CSQ-8), with 1 of 15 matched pairs reaching p < .05 — which is "
        "what chance produces.",
        fill=CWASH, edge=CAVEAT, label_color=CAVEAT, size=12.5)
provenance(s, ["meetings/build/_exp2_summary.md §5, §6"])

# ── 14 · the negative result ──────────────────────────────────────────────────
s, y = d.newslide("Act II · Exp2", "The finding that shaped Exp3: two instruments did not train")
y = band(s, y, "THE NEGATIVE RESULT",
         "Train PTO on WAI-SR and WAI-SR does not improve. Train it on CSQ-8 and CSQ-8 does not "
         "improve. Only Q1+Q2 responds to being optimised.",
         fill=CWASH, edge=CAVEAT, label_color=CAVEAT)
rows = [
    ["Q1+Q2", "Q1+Q2", "2.378", "2.968", "+0.590", "<.0001"],
    ["WAI-SR", "WAI-SR total", "2.708", "2.788", "+0.081", ".414"],
    ["CSQ-8", "CSQ-8", "2.182", "2.255", "+0.073", ".304"],
]
table(s, y + Inches(0.08),
      ["trained on", "measured on", "base", "best", "Δ", "p"],
      rows, col_w=[Inches(2.0), Inches(2.35), Inches(1.5), Inches(1.5), Inches(1.5),
                   Inches(1.35)],
      emphasis=lambda i, j, v: i > 0 and j in (4, 5))
ty = table_bottom(y + Inches(0.08), 3)
ty = caption(s, ty, "Each arm scored on its OWN training instrument — the fairest possible test of "
                    "whether the optimisation took.  n = 96 paired, uncorrected.")
bullets(s, ty + Inches(0.02), [
    ("Why this matters for the thesis.", "It is the reason RQ-iii — “does the oracle questionnaire "
     "change the conclusion?” — is still open rather than answered. Exp2 tried, and two of three "
     "instruments produced no training signal worth measuring."),
    ("It is also why Exp3 trains on Q1+Q2 only,", "and reads all 8 instruments at evaluation. "
     "Choosing what to READ is free; choosing what to TRAIN ON costs a full run each."),
], size=12.5, gap=9)
provenance(s, ["meetings/build/_exp2_summary.md §5  ←  Exp2_PTO/eda/eval/{WAI_SR,CSQ8}/**"])

# ── 15 · the comparability warning ────────────────────────────────────────────
s, y = d.newslide("Act II · Exp2", "Why no Exp2 number appears on an Exp3 axis")
y = band(s, y, "THE STANDING WARNING",
         "Exp2 scores 2.378 at base and Exp3 scores 3.008 — same base model, same oracle. The gap "
         "is generation precision: Exp2 ran 4-bit NF4, Exp3 bf16.",
         fill=CWASH, edge=CAVEAT, label_color=CAVEAT, size=14.5)
rows = [
    ["Degenerate therapist turns", "8.73%", "0.34%", "reproduces  (claimed ≈9.5% vs ≈0.3%)"],
    ["Base Q1+Q2, all conversations", "2.378", "3.008", "reproduces  (claimed 2.38 vs ≈3.0)"],
    ["Base Q1+Q2, non-degenerate only", "2.648", "3.022", "does NOT  (claimed ≈2.93 ≈ parity)"],
]
table(s, y + Inches(0.06), ["measured", "Exp2", "Exp3", "vs what the docs claim"], rows,
      col_w=[Inches(3.5), Inches(1.5), Inches(1.5), Inches(5.0)], prose_cols=(3,),
      emphasis=lambda i, j, v: i == 2)
ty = table_bottom(y + Inches(0.06), 3)
bullets(s, ty + Inches(0.06), [
    ("Removing degenerate conversations closes 41% of the gap, not all of it.", "A clean-vs-clean "
     "gap of +0.374 survives every filter tried. The conclusion — never compare levels across "
     "experiments — is unchanged and if anything stronger."),
    ("But the stated MECHANISM is weaker than advertised.", "Under the repo's own loop detector "
     "the two experiments are 1.4× apart, not 30×. And a second confound was never mentioned: "
     "Exp2 stopped generation on one ChatML marker where Exp3 stops on two and post-cleans, "
     "leaving 2.2% of Exp2 turns carrying raw control markers against 0% in Exp3."),
], size=12.5, gap=9)
provenance(s, ["recomputed from Exp2_PTO/data/conversations_eval/** and "
               "Exp3_PTO_GRPO/data/*/conversations/full/**"])

# ── 16 · Act II close ─────────────────────────────────────────────────────────
s, y = d.newslide("Act II · Exp2", "What Exp2 settled, and what it handed forward")
half = CW / 2 - Inches(0.16)
rect(s, ML, y, half, Inches(3.72), VWASH)
rect(s, ML, y, Inches(0.055), Inches(3.72), VERDICT)
tfa = txbox(s, ML + Inches(0.3), y + Inches(0.22), half - Inches(0.6), Inches(3.9))
run(para(tfa, first=True), "SETTLED", size=9.5, bold=True, color=VERDICT)
for t in ["A 1B therapist can be trained by PTO. The Exp1 result was not a property of model size.",
          "It survives harder patients and a schema-constrained oracle.",
          "Which instrument you TRAIN on matters enormously — two of three produced nothing.",
          "Absolute scores are not portable across experiments. Report within-experiment "
          "contrasts only."]:
    pp = para(tfa, space_before=13)
    run(pp, t, size=12.5, color=INK)

rect(s, ML + half + Inches(0.32), y, half, Inches(3.72), CWASH)
rect(s, ML + half + Inches(0.32), y, Inches(0.055), Inches(3.72), CAVEAT)
tfb = txbox(s, ML + half + Inches(0.62), y + Inches(0.22), half - Inches(0.6), Inches(3.9))
run(para(tfb, first=True), "HANDED FORWARD", size=9.5, bold=True, color=CAVEAT)
for t in ["Four things changed at once, so nothing here attributes to a cause.",
          "Look-ahead did nothing at matched iterations. Is the Exp1 K result regime-specific, "
          "or was Exp2 simply too short and too confounded to see it?",
          "Still one model as both patient and oracle. Still no independent measurement.",
          "PTO has never been compared to anything. Beating an untrained baseline is a low bar — "
          "how does it fare against a real alternative optimiser?"]:
    pp = para(tfb, space_before=13)
    run(pp, t, size=12.5, color=INK)


# ══════════════════════════════════════════════════════════════════════════════
# ACT III — Exp3
# ══════════════════════════════════════════════════════════════════════════════
d.divider("ACT III", "Exp3 — the controlled comparison",
          "One lever at a time. PTO and GRPO, K ∈ {0, 5}, matched MCL, matched candidate budget "
          "(M = G = 8), matched temperature, one oracle — and a second grader held out.",
          accent=GRPO_C)

# ── 18 · the grid ─────────────────────────────────────────────────────────────
s, y = d.newslide("Act III · Exp3", "The design, and the two things Exp1 and Exp2 lacked")
left_w = CW * 0.52
bullets(s, y, [
    ("A comparator.", "GRPO — group-relative optimisation. It slices an on-policy rollout and "
                      "scores G completions per prompt. PTO grows a best-of-M reranked trunk. "
                      "Same candidate budget, same temperature, same oracle."),
    ("An independent grader.", "Claude Haiku 4.5. Different model family, never played the "
                              "patient, never was the training reward. Every state scored twice."),
    ("Everything else held fixed.", "MCL = 12, bf16, Q1+Q2 as the only training reward, all 8 "
                                    "instruments read at evaluation."),
], size=13, gap=12, width=left_w)
bx = ML + left_w + Inches(0.4)
bw = CW - left_w - Inches(0.4)
rect(s, bx, y, bw, Inches(3.5), WASH)
tfb = txbox(s, bx + Inches(0.28), y + Inches(0.24), bw - Inches(0.56), Inches(3.05))
run(para(tfb, first=True), "THE GRID", size=9.5, bold=True, color=MUTED)
for lead, rest in [
    ("4 arms", "PTO / GRPO × K ∈ {0, 5}"),
    ("40 model states", "11 + 11 + 11 + 7, base included"),
    ("2 graders", "every state scored on both"),
    ("30,720 cells", "40 × 8 rubrics × 96 personas, per grader"),
]:
    pp = para(tfb, space_before=13)
    run(pp, lead, size=15, bold=True, color=INK)
    run(pp, "   " + rest, size=11.5, color=BODY)
pn = para(tfb, space_before=16)
run(pn, "Complete except GRPO K=5, which stopped at iteration 6 — slide 26.",
    size=10.5, italic=True, color=CAVEAT)
provenance(s, ["measurement/validity/tables/multijudge_coverage.md"])

# ── 19 · RQ-i ─────────────────────────────────────────────────────────────────
s, y = d.newslide("Act III · RQ-i · does looking ahead help?",
                  "The lever works — in opposite directions")
y = band(s, y, "VERDICT",
         "Look-ahead HELPS GRPO and HURTS PTO. At iteration 6 both effects are significant, on "
         "both graders.",
         fill=VWASH, edge=VERDICT, label_color=VERDICT)
rows = [
    ["4", "+0.120  (0.20)", "+0.123  (0.21)", "−0.115  (−0.25) *", "−0.233  (−0.37) **"],
    ["5", "−0.002  (−0.00)", "+0.173  (0.33) *", "−0.070  (−0.13)", "−0.311  (−0.43) **"],
    ["6", "+0.257  (0.42) ***", "+0.343  (0.51) ***", "−0.263  (−0.42) ***", "−0.533  (−0.55) ***"],
]
table(s, y + Inches(0.05),
      ["iteration", "PTO · primary", "PTO · held-out", "GRPO · primary", "GRPO · held-out"],
      rows, col_w=[Inches(1.5), Inches(2.5), Inches(2.5), Inches(2.5), Inches(2.5)],
      emphasis=lambda i, j, v: i == 2)
ty = table_bottom(y + Inches(0.05), 3)
caption(s, ty, "Q1+Q2, persona-paired mean difference (Cohen's dz).  Sign: + = K=0 higher.  "
               "* p_holm < .05   ** < .01   *** < .001")
bandbot(s, "AND IT REVERSES EXP1",
        "Exp1 found K=5 above K=0 with PTO. Exp3 finds the opposite for PTO — and finds Exp1's "
        "direction only for GRPO, an optimiser Exp1 never ran. The lever's sign is not a property "
        "of look-ahead.",
        fill=CWASH, edge=CAVEAT, label_color=CAVEAT, size=12.5)
provenance(s, ["lookahead/reward/tables/k_table1.md"])

# ── 20 · RQ-i figure ──────────────────────────────────────────────────────────
s, y = d.newslide("Act III · RQ-i · does looking ahead help?",
                  "The same lever, both optimizers, both graders")
figband(s, y, fig("lookahead", "reward", "figures", "k_contrast_both_judges.png"), "READ IT AS",
        "Two panels that mirror each other. Whatever look-ahead is doing, it is not a property of "
        "the lever alone — it is an interaction with the optimizer.",
        fill=WASH, edge=PTO_C, label_color=MUTED, size=12.5, bold_text=False)
provenance(s, [_rel(fig("lookahead", "reward", "figures", "k_contrast_both_judges.png"))])

# ── 21 · the DiD ──────────────────────────────────────────────────────────────
s, y = d.newslide("Act III · RQ-i · does looking ahead help?",
                  "The interaction is significant on both graders")
y = band(s, y, "VERDICT",
         "The K × optimizer interaction now reaches significance on the PRIMARY oracle too, not "
         "only on the held-out judge.",
         fill=VWASH, edge=VERDICT, label_color=VERDICT)
rows = [
    ["4", "+0.235", "0.286", ".104   (ns)", "+0.356", "0.401", ".008"],
    ["5", "+0.068", "0.095", "1.000  (ns)", "+0.484", "0.525", "<.001"],
    ["6", "+0.520", "0.605", "<.001", "+0.876", "0.754", "<.001"],
]
table(s, y + Inches(0.05),
      ["iter", "DiD · primary", "dz", "p_holm", "DiD · held-out", "dz", "p_holm"],
      rows, col_w=[Inches(1.0), Inches(1.95), Inches(1.0), Inches(1.35), Inches(1.95),
                   Inches(1.0), Inches(1.35)],
      emphasis=lambda i, j, v: i == 2)
ty = table_bottom(y + Inches(0.05), 3, pad=Inches(0.14))
bullets(s, ty, [
    ("It is not a single-iteration artefact.", "On the held-out judge dz runs 0.167, 0.090, 0.401, "
     "0.525, 0.754 over iterations 2 to 6 — it dips at 3, then rises steadily. Iteration 0 — four "
     "independent draws of the same untrained policy — sits at −0.033, this design's noise floor."),
    ("It holds on every instrument but one.", "At iteration 6 the interaction is Holm-significant "
     "under both graders on all eight rubrics except MI-inconsistency."),
], size=12.5, gap=7)
bandbot(s, "RETIRED",
        "“The grader that WAS the training reward cannot see this effect.” True through iteration "
        "5, false at 6. The defensible claim is now that it sees it 1.7× less sharply "
        "(0.876 / 0.520).",
        fill=CWASH, edge=CAVEAT, label_color=CAVEAT, size=12.5)
provenance(s, ["lookahead/reward/tables/k_did.md"])

# ── 22 · retention ────────────────────────────────────────────────────────────
s, y = d.newslide("Act III · RQ-i · does looking ahead help?",
                  "Does the gain survive a grader that was never the reward?")
y = band(s, y, "VERDICT",
         "On GRPO, K=5 keeps 84% of its measured gain under the held-out judge against K=0's 57%, "
         "and the intervals are disjoint.",
         fill=VWASH, edge=VERDICT, label_color=VERDICT)
rows = [
    ["GRPO", "6", "0.567", "[0.379, 0.728]", "0.844", "[0.745, 0.959]", "yes"],
    ["PTO", "10", "0.823", "[0.720, 0.947]", "0.639", "[0.551, 0.738]", "NO — overlapping"],
]
table(s, y + Inches(0.1),
      ["method", "iter", "K=0 retention", "95% CI", "K=5 retention", "95% CI", "disjoint?"],
      rows, col_w=[Inches(1.35), Inches(0.8), Inches(1.8), Inches(2.0), Inches(1.8),
                   Inches(2.0), Inches(1.85)],
      emphasis=lambda i, j, v: i == 0 and j in (4, 6))
ty = table_bottom(y + Inches(0.1), 2)
bullets(s, ty, [
    ("Retention", "= gain seen by the held-out judge ÷ gain seen by the training oracle. 1.0 means "
                  "the improvement is fully real to a grader with no stake in it; a low number is "
                  "a gain the training oracle believes and an independent grader does not."),
    ("Read the PTO row as directional only.", "0.823 vs 0.639 looks like a clean separation and is "
     "not — the intervals overlap by 0.018. It IS disjoint on Q2 and MITI, so name the metric."),
], size=12.5, gap=8)
bandbot(s, "NOT THIS",
        "Not that PTO reward-hacks and GRPO does not. PTO's K=0 arm retains 0.823 — higher than "
        "either GRPO arm. The contrast is WITHIN method, across K.",
        fill=CWASH, edge=CAVEAT, label_color=CAVEAT, size=12.5)
provenance(s, ["lookahead/transfer/tables/k_retention_summary.md"])

# ── 23 · RQ-ii at K=0 ─────────────────────────────────────────────────────────
s, y = d.newslide("Act III · RQ-ii · which optimizer?", "At K = 0, PTO wins clearly")
y = band(s, y, "VERDICT",
         "At the matched 10-iteration endpoint with no look-ahead, PTO beats GRPO on both graders, "
         "with a large effect on the held-out judge.",
         fill=VWASH, edge=VERDICT, label_color=VERDICT)
rows = [
    ["Q1+Q2", "+0.507", "0.729", "<.001", "+0.609", "1.265", "<.001"],
    ["MITI  (MI integrity)", "+0.352", "0.459", "<.001", "+0.253", "0.648", "<.001"],
    ["MICI  (inconsistency) ↓", "−0.346", "−0.989", "<.001", "−0.225", "−0.667", "<.001"],
]
table(s, y + Inches(0.1), ["metric", "Δ primary", "dz", "p_holm", "Δ held-out", "dz", "p_holm"],
      rows, col_w=[Inches(2.9), Inches(1.55), Inches(1.0), Inches(1.25), Inches(1.55),
                   Inches(1.0), Inches(1.25)],
      emphasis=lambda i, j, v: i == 0)
ty = table_bottom(y + Inches(0.1), 3)
bullets(s, ty, [
    ("Sign.", "Δ = PTO − GRPO. Positive favours PTO; MICI is lower-is-better, so its negative Δ "
              "also favours PTO."),
    ("Mechanism.", "GRPO K=0 peaks at iteration 8 (4.082) then regresses to 3.753 while its "
                   "MI-inconsistency climbs to 0.838. PTO climbs steadily to 4.260."),
], size=12.5, gap=8)
provenance(s, ["method/contrast/tables/method_paired_by_K.md",
               "arms/outcomes/tables/<judge>/leaderboard_scorecard.md"])

# ── 24 · RQ-ii at K=5 ─────────────────────────────────────────────────────────
s, y = d.newslide("Act III · RQ-ii · which optimizer?", "At K = 5, the answer flips")
y = band(s, y, "VERDICT",
         "Turn look-ahead on and GRPO wins. At matched iteration 6 it beats PTO on both graders; "
         "best-vs-best it ties on the primary and wins on the held-out judge.",
         fill=VWASH, edge=VERDICT, label_color=VERDICT)
rows = [
    ["matched iteration 6", "−0.332", "−0.437", "<.001", "−0.397", "−0.599", "<.001"],
    ["best vs best", "+0.078", "0.133", "1.000 (ns)", "−0.168", "−0.352", ".009"],
]
table(s, y + Inches(0.08), ["comparison", "Δ primary", "dz", "p_holm", "Δ held-out", "dz",
                            "p_holm"],
      rows, col_w=[Inches(2.9), Inches(1.55), Inches(1.0), Inches(1.3), Inches(1.55),
                   Inches(1.0), Inches(1.3)],
      emphasis=lambda i, j, v: True)
ty = table_bottom(y + Inches(0.08), 2)
bullets(s, ty, [
    ("Δ = PTO − GRPO,", "so every negative number on this slide favours GRPO."),
    ("So RQ-i and RQ-ii are not separable.", "“Which optimizer is better” has no answer that is "
     "not conditional on K, and “does look-ahead help” has none that is not conditional on the "
     "optimizer. That is the central finding of the thesis."),
], size=12.5, gap=8)
bandbot(s, "NOT THIS",
        "Not that GRPO overtakes PTO given enough iterations. GRPO K=5 has SIX; PTO has ten. The "
        "flip is at matched iteration, and the arm is censored.",
        fill=CWASH, edge=CAVEAT, label_color=CAVEAT, size=12.5)
provenance(s, ["method/contrast/tables/method_paired_{by_K,best}.md"])

# ── 25 · compute ──────────────────────────────────────────────────────────────
s, y = d.newslide("Act III · RQ-ii · which optimizer?", "An iteration is not a unit of spend")
y = band(s, y, "VERDICT",
         "On the compute axis PTO dominates outright: it reaches iteration 10 for 8.1 GPU-h "
         "against GRPO's 27.9 — 3.4× cheaper, and it scores higher.",
         fill=VWASH, edge=VERDICT, label_color=VERDICT)
rows = [
    ["PTO  K=0", "10", "8.12", "0.81", "4.260", "2.866"],
    ["PTO  K=5", "10", "19.68", "1.97", "4.307", "2.667"],
    ["GRPO K=0", "10", "27.91", "2.79", "3.753", "2.257"],
    ["GRPO K=5", "6  (+ partial 7)", "30.53", "5.09", "4.229", "2.903"],
]
table(s, y + Inches(0.1),
      ["arm", "iterations", "GPU-h", "GPU-h / iter", "Q1+Q2 primary", "Q1+Q2 held-out"],
      rows, col_w=[Inches(1.9), Inches(2.0), Inches(1.4), Inches(1.8), Inches(2.2), Inches(2.2)],
      emphasis=lambda i, j, v: (i == 0 and j in (2, 3)) or (i == 3 and j == 5))
ty = table_bottom(y + Inches(0.1), 4)
bullets(s, ty, [
    ("Why PTO is cheap.", "Its dominant phase is building preference pairs — 5.7 of 8.1 h — which "
                          "does not scale with the optimizer step count. GRPO computes its reward "
                          "inside the training loop, so every step pays."),
    ("Look-ahead costs ~1.9× per optimizer step,", "measured as median ratios 1.97 / 1.96 / 1.91 "
     "at iterations 3 / 4 / 5. The two GRPO arms are no longer budget-matched: K=5 is 9.4% more "
     "expensive at its last adapter."),
], size=12.5, gap=8)
provenance(s, ["compute/cost/tables/{compute_by_arm,compute_by_iteration,step_multiplier}.md"])

# ── 26 · RQ-iii, now with Exp2 evidence ───────────────────────────────────────
s = d._next()
rect(s, 0, 0, W, H, INK)
tf = txbox(s, ML, Inches(1.25), CW, Inches(1.4))
run(para(tf, first=True), "RQ-iii", size=15, bold=True, color=GOLD)
p = para(tf, space_before=12)
run(p, "Does the oracle questionnaire change the conclusion?", size=34, bold=True, color=PAPER)
rect(s, ML, Inches(2.92), CW, Inches(3.35), PANEL)
tfb = txbox(s, ML + Inches(0.42), Inches(3.2), CW - Inches(0.84), Inches(2.8))
run(para(tfb, first=True), "HELD — AND EXP2 IS WHY", size=9.5, bold=True, color=GOLD)
pb = para(tfb, space_before=11)
run(pb, "Exp3 trains every arm on Q1+Q2 and EVALUATES on all 8 instruments — so we can already "
        "say the conclusions do not depend on which rubric we READ.", size=14.5, color=PAPER)
pb = para(tfb, space_before=11)
run(pb, "Whether they depend on which rubric we TRAIN ON is the open half. Exp2 is the only "
        "experiment that tried it, across three instruments — and two of the three produced no "
        "measurable gain on their own metric (WAI-SR p = .414, CSQ-8 p = .304).", size=14.5,
    color=MIST)
pb = para(tfb, space_before=11)
run(pb, "So the honest position is not “untested”. It is “tested once, in a confounded regime, "
        "with a mostly negative result”. Redoing it properly means four more Exp3 arms at "
        "8–28 GPU-h each.", size=14.5, color=MIST)
pb = para(tfb, space_before=13)
run(pb, "Recommendation: keep it held. RQ-i and RQ-ii are the thesis.", size=14, bold=True,
    color=MINT)

# ── 27 · measurement ──────────────────────────────────────────────────────────
s, y = d.newslide("Act III · can we trust any of this?", "The measurement thread, in one slide")
y = band(s, y, "VERDICT",
         "An independent grader in a different model family reproduces 88.5% of all 6,240 "
         "arm × metric contrasts — and 99.5% of the large ones.",
         fill=VWASH, edge=VERDICT, label_color=VERDICT)
left_w = CW * 0.44
rows = [
    ["all contrasts", "6,240", "88.5%"],
    ["|Δ| ≥ 0.10", "4,035", "94.5%"],
    ["|Δ| ≥ 0.25", "2,539", "97.4%"],
    ["|Δ| ≥ 0.50", "1,309", "99.5%"],
]
table(s, y + Inches(0.1), ["subset", "n", "same sign"], rows,
      col_w=[Inches(2.2), Inches(1.4), Inches(1.7)], left=ML)
bullets(s, table_bottom(y + Inches(0.1), 4, pad=Inches(0.1)),
        [("6,240", "= 8 metrics × C(40, 2) = 8 × 780.")],
        size=11, gap=4, width=left_w, x=ML)
bx = ML + left_w + Inches(0.35)
bw = CW - left_w - Inches(0.35)
bullets(s, y + Inches(0.02), [
    ("Oracle repeatability.", "ICC(2,1) 0.86–0.99 across four re-draws of the same conversations."),
    ("Dependability, one judge.", "Strong for Q1/Q2/WAI-SR/CSQ-8/PCT (0.91–0.96) and MICI (0.85). "
                                  "MITI is the weak instrument at 0.62 — name it whenever a "
                                  "channel result rests on MITI."),
    ("This is what Exp1 and Exp2 never had.", "In both, the model that produced the reward was "
                                              "also the model that graded the result. Every "
                                              "held-out number in this deck exists only in Exp3."),
], size=12.5, gap=10, width=bw, x=bx)
provenance(s, ["measurement/validity/tables/multijudge_{sign_preservation,variance_components}.md"])

# ── 28 · the stalled arm ──────────────────────────────────────────────────────
s, y = d.newslide("Act III · Exp3", "One honest slide about the run that stopped")
y = band(s, y, "WHAT HAPPENED",
         "GRPO K=5's iteration 7 needs 106 optimizer steps. 40 are on disk. Across four Colab "
         "sessions, 132 further steps were computed and then thrown away.",
         fill=CWASH, edge=CAVEAT, label_color=CAVEAT)
rows = [
    ["1", "19 Aug", "1 → 103", "1 → 30", "writes to Drive stopped mid-save"],
    ["2", "20 Aug", "—", "—", "OpenAI org spend limit (384 of 395 log lines)"],
    ["3", "20 Aug", "—", "—", "OpenAI org spend limit"],
    ["4", "20 Aug", "31 → 99", "31 → 40", "writes to Drive stopped mid-save"],
]
table(s, y + Inches(0.05), ["session", "date", "steps trained", "steps saved", "outcome"], rows,
      col_w=[Inches(1.1), Inches(1.1), Inches(1.7), Inches(1.6), Inches(5.4)],
      emphasis=lambda i, j, v: j == 2, prose_cols=(4,))
bullets(s, table_bottom(y + Inches(0.05), 4), [
    ("Training was never the problem.", "Steps after the stall ran at 162.1 s against 161.1 s "
     "before it. GPU and oracle were healthy throughout — only new files stopped reaching Drive."),
    ("It is not chronic.", "All 16 previously completed iterations have exactly as many saved "
     "artifacts as optimizer steps. Iteration 7 is the only anomaly."),
    ("The fix is operational.", "Write checkpoints to local Colab disk and copy to Drive once per "
     "iteration. Nothing in the experiment name, the reward or the data changes."),
], size=12, gap=7)

# ── 29 · threats ──────────────────────────────────────────────────────────────
s, y = d.newslide("Threats", "Where the evidence is thin — said out loud")
bullets(s, y, [
    ("Every contested endpoint is a single 96-conversation draw.", "The only noise floor is at the "
     "base: 4 independent draws of the identical base policy give 54 same-policy contrasts, 0 of "
     "which reach even uncorrected p < .05 (max |dz| 0.128). Reassuring for the base; silent "
     "about a trained checkpoint."),
    ("GRPO K=5 is censored at 6.", "Every statement about it is “within six iterations”. Whether "
     "its lead persists, or it regresses the way GRPO K=0 did after iteration 8, is unobserved."),
    ("MITI dependability is 0.62 off one judge,", "and MITI carries the MI-integrity channel "
     "results. There is no channel-level ICC at all."),
    ("All 96 personas are used for both training and evaluation", "at every iteration, so every "
     "number is in-sample with respect to the patient distribution."),
    ("Exp1 and Exp2 have no held-out grader at all,", "so their internal contrasts cannot be "
     "checked the way Exp3's can. Act I's re-scoring is one additional grader on the transcripts "
     "— not a second draw, and not a second experiment."),
], size=12.5, gap=11)
provenance(s, ["results/LIMITATIONS.md"])

# ── 30 · what the three together establish ────────────────────────────────────
s, y = d.newslide("The arc", "What the three experiments establish together")
rows = [
    ["PTO improves a therapist LLM", "yes  ✓", "yes  ✓", "yes  ✓", "replicated 3×, 2 model sizes"],
    ["…and it survives an independent grader", "—", "—", "yes  ✓", "Exp3 only"],
    ["Look-ahead helps PTO", "weakly", "wash", "no  ✗", "sig. on 1 of 4 metrics in Exp1; null in "
                                                        "Exp2; reversed in Exp3"],
    ["Look-ahead helps GRPO", "—", "—", "yes  ✓", "the direction Exp1 saw, wrong method"],
    ["The training instrument matters", "—", "yes  ✓", "open", "2 of 3 Exp2 arms did not train"],
    ["PTO beats a real alternative", "—", "—", "at K=0", "at K=5 it loses"],
]
table(s, y + Inches(0.04), ["claim", "Exp1", "Exp2", "Exp3", "reading"], rows,
      col_w=[Inches(4.0), Inches(1.15), Inches(1.15), Inches(1.15), Inches(4.15)],
      prose_cols=(4,), row_h=Inches(0.42),
      emphasis=lambda i, j, v: i in (0, 1) and j == 3)
bullets(s, table_bottom(y + Inches(0.04), 6, row_h=Inches(0.42), pad=Inches(0.14)), [
    ("The one claim that strengthened across all three experiments is the plainest one.",
     "PTO works. It replicates across two therapist model families, two patient regimes, two "
     "oracles, and — only in Exp3 — a grader with no stake in the outcome."),
    ("The one that dissolved is the headline of the published paper.",
     "Look-ahead is not a knob with a sign. It helps group-relative optimisation and hurts "
     "preference-tree optimisation, and Exp1 could not have seen that because it had only "
     "one optimiser."),
], size=12, gap=8)

# ── 31 · decisions ────────────────────────────────────────────────────────────
s, y = d.newslide("Decisions", "Three things I would like decided today")
y += Inches(0.02)
for num, q, col, body, rec in [
    ("1", "Finish GRPO K=5?", GRPO_C,
     "Resume iteration 7 and run to 10: ~16–18 GPU-h, ~$85–130. The arm lands near 48–50 GPU-h "
     "against its K=0 sibling's 28.",
     "Recommendation: yes — it is the only arm without an endpoint, and it is currently the "
     "best-scoring arm on the held-out judge."),
    ("2", "How much of Act I goes in the thesis?", EXP1_C,
     "The re-audit found the published ANOVA was run on 3 groups of 15, and the depth contrast is "
     "significant on one metric of four. The re-scoring supports the direction but not the "
     "strength.",
     "Recommendation: include it. A thesis that re-examines its own prior work is stronger than "
     "one that cites it — and Exp3 reverses the sign for PTO regardless."),
    ("3", "Buy the replicate draw?", VERDICT,
     "A second independent 96-conversation draw of ~5 contested endpoints: ~$10 and about one "
     "A100-hour, or four free hours locally. No code change needed.",
     "Recommendation: yes — it either retires the endpoint-fragility objection thesis-wide or "
     "tells us the headline is fragile."),
]:
    hgt = Inches(1.62)
    rect(s, ML, y, CW, hgt, WASH)
    rect(s, ML, y, Inches(0.055), hgt, col)
    tfn = txbox(s, ML + Inches(0.32), y + Inches(0.16), Inches(0.6), Inches(0.6))
    run(para(tfn, first=True), num, size=22, bold=True, color=col)
    tfq = txbox(s, ML + Inches(1.0), y + Inches(0.13), CW - Inches(1.5), Inches(1.45))
    run(para(tfq, first=True), q, size=15.5, bold=True, color=INK)
    pb = para(tfq, space_before=5)
    run(pb, body, size=11.5, color=BODY)
    pr = para(tfq, space_before=5)
    run(pr, rec, size=11.5, bold=True, color=col)
    y += hgt + Inches(0.16)

# ── 32 · in one line ──────────────────────────────────────────────────────────
s = d._next()
rect(s, 0, 0, W, H, INK)
tf = txbox(s, ML, Inches(1.85), CW, Inches(3.9))
run(para(tf, first=True), "IN ONE LINE", size=11, bold=True, color=EXP1_C)
p = para(tf, space_before=18)
run(p, "Look-ahead is not a knob with a sign.", size=32, bold=True, color=PAPER)
p = para(tf, space_before=8)
run(p, "It helps group-relative optimization and hurts preference-tree optimization,",
    size=23, color=SKY)
p = para(tf, space_before=4)
run(p, "and which optimizer you should prefer depends on whether you turned it on.",
    size=23, color=SKY)
p = para(tf, space_before=24)
run(p, "Three experiments agree that PTO works. Only the third could have found that the lever "
       "Exp1 was built around points the other way for the method Exp1 used.",
    size=14, color=SLATE)
p = para(tf, space_before=10)
run(p, "PTO remains 3.4× cheaper and wins outright at K = 0.", size=14, color=SLATE)

# ── 33-34 · appendix: the two frameworks ──────────────────────────────────────
for name, png, col, note in [
    ("PTO — preference tree + DPO", "pto_framework.png", PTO_C,
     "Branch M candidates per therapist turn, look ahead K, score, keep (chosen, rejected) where "
     "the margin clears τ, then a DPO update. This is the method Exp1, Exp2 and Exp3 all run."),
    ("GRPO — group-relative rollout", "grpo_framework.png", GRPO_C,
     "Slice the rollout after every patient turn, sample G completions per prompt, score all of "
     "them, use the group-relative advantage. The oracle sits INSIDE the update. Exp3 only."),
]:
    s, y = d.newslide("Appendix", name)
    figband(s, y, fig("schematics", png), "IN ONE LINE", note,
            fill=WASH, edge=col, label_color=MUTED, size=12.5, bold_text=False)
    provenance(s, ["schematics/" + png])

# ── 35 · appendix: trajectories ───────────────────────────────────────────────
s, y = d.newslide("Appendix", "All four Exp3 arms, held-out judge")
figband(s, y, fig("arms", "outcomes", "figures", "claude-haiku-4-5", "trajectories",
                  "trajectory_Q1Q2.png"), "READ IT AS",
        "GRPO K=0 (orange) is non-monotonic — 2.637 at iteration 3, back to 2.617 at 8, then down "
        "to 2.257 by 10. GRPO K=5 climbs to 2.903 in six iterations, the highest last-scored value "
        "of any arm on this grader — but it stopped there rather than finishing.",
        fill=WASH, edge=GRPO_C, label_color=MUTED, size=12.5, bold_text=False)
provenance(s, [_rel(fig("arms", "outcomes", "figures", "claude-haiku-4-5", "trajectories",
                        "trajectory_Q1Q2.png"))])

# ── 36 · appendix: compute trajectory ─────────────────────────────────────────
s, y = d.newslide("Appendix", "GPU-hours per iteration, reconstructed from artifact mtimes")
pic(s, fig("compute", "cost", "figures", "compute_trajectory.png"), ML, y + Inches(0.02), CW,
    H - y - Inches(1.05))
provenance(s, [_rel(fig("compute", "cost", "figures", "compute_trajectory.png"))])


d.save(OUT, repo=REPO)
