"""Print the two complete process tables (Appendix A: training oracle; Appendix B: held-out judge).

Lior's read, 2026-09-30: "Table 3: do we have a full table of all iterations in the appendix?" --
there was none, only figures. These tables give Table 3's twelve measures at every model state
(the shared Base, then K=0 and K=5 at iterations 1-10), one table per judge, laid out like the
complete score tables (Tables 4 and 6).

**Nothing here computes a number.** Every cell is read from the EDA's tracked workbook
``lookahead/shared_base/tables/shared_base.xlsx``:

- levels: ``process_levels_<judge>`` (the eight turn-share and reply measures, and ``ct_prop``)
  and ``persist_levels_<judge>`` (``ct_persist_mean``, ``st_to_ct_mean``);
- stars: ``k_process_paired`` and ``k_persistence`` -- the persona-paired K=5 vs K=0 contrast at
  each iteration, Holm across iterations 1-10 within measure; a star goes on the cell of the
  ``better`` run (the sheets already account for lower-is-better measures) wherever ``p_holm``
  < .05, as in Tables 4 and 6.

Before printing, every iteration-10 cell is checked against Table 3 / Table 7 as they stand in the
.tex (levels to three decimals; Table 3's bold must be each row's best printed value of the Base,
K=0 and K=5; its dz cells and stars must match the iteration-10 test), so the new tables cannot
drift from the body.

    & ..\\..\\.venv\\Scripts\\python.exe render_process_tables.py            # print both tables
    & ..\\..\\.venv\\Scripts\\python.exe render_process_tables.py --codes    # the code-mix table

``--codes`` (since 2026-10-05) prints the rows of the Appendix A code-mix table instead: every
one of the coder's eleven therapist codes at the Base and at iteration 10, both judges, with the
iteration-10 paired dz from ``k_process_paired`` (the same test and stars as Table 3).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
SHARED_BASE_XLSX = (HERE.parent.parent / "Exp3_PTO_GRPO" / "eda" / "results" / "lookahead"
                    / "shared_base" / "tables" / "shared_base.xlsx")
SECTIONS = HERE / "sections"

# (column key, source, metric in the level sheet, metric in the paired sheet, Table 3 row label
# exactly as printed, lower is better)
DN = r"($\downarrow$)"
COLS = [
    ("pra", "proc", "th_PRA_rate", "th_PRA_rate", f"non-specific praise {DN}", True),
    ("cr", "proc", "th_CR_rate", "th_CR_rate", "complex reflection", False),
    ("pers", "proc", "th_PERS_rate", "th_PERS_rate", f"persuasion {DN}", True),
    ("adh", "proc", "mi_adherent_rate", "mi_adherent_rate", "MI-consistent share", False),
    ("inc", "proc", "mi_incons_rate", "mi_incons_rate", f"MI-inconsistent share {DN}", True),
    ("refl_ct", "proc", "refl_after_ct", "refl_after_ct", "change talk: reflects it", False),
    ("pra_st", "proc", "pra_after_st", "pra_after_st", f"sustain talk: praises {DN}", True),
    ("pers_st", "proc", "pers_after_st", "pers_after_st", f"sustain talk: persuades {DN}", True),
    ("refl_st", "proc", "refl_after_st", "refl_after_st", "sustain talk: reflects it", False),
    ("ct_ct", "persist", "ct_persist_mean", "ct_persist", "change talk", False),
    ("st_ct", "persist", "st_to_ct_mean", "st_to_ct", "sustain talk", False),
    ("ct_prop", "proc", "ct_prop", "ct_prop", "change-talk share", False),
]
JUDGES = {"gpt-4o-mini": ("07_patient.tex", "tab:process"),
          "claude-haiku-4-5": ("A2_heldout.tex", "tab:process-heldout")}
ARMS = {"GRPO_LA0": "K0", "GRPO_LA5": "K5"}


def load(judge: str):
    proc = pd.read_excel(SHARED_BASE_XLSX, sheet_name=f"process_levels_{judge}")
    pers = pd.read_excel(SHARED_BASE_XLSX, sheet_name=f"persist_levels_{judge}")
    paired = pd.concat([pd.read_excel(SHARED_BASE_XLSX, sheet_name=s)
                        for s in ("k_process_paired", "k_persistence")])
    paired = paired[(paired.judge == judge) & (paired.method == "GRPO")]
    return proc, pers, paired


def level(proc, pers, arm: str, it: int, src: str, col: str) -> float:
    d = proc if src == "proc" else pers
    r = d[(d.arm == arm) & (d.iteration == it)]
    assert len(r) == 1, (arm, it, col)
    return float(r.iloc[0][col])


def table3_rows(tex_name: str, label: str) -> dict[str, list[str]]:
    """The rows of Table 3 / Table 8 as printed: label -> [Base, K0 at best, K0 at 10, K5 at 10]
    strings ('B' prefix = bold). Since 2026-10-07 the two tables carry both anchors."""
    tex = (SECTIONS / tex_name).read_text(encoding="utf-8")
    body = tex[:tex.index(f"\\label{{{label}}}")]
    body = body[body.rindex("\\begin{tabular}"):]
    rows = {}
    for line in body.splitlines():
        cells = re.findall(r"\$(\\mathbf\{)?([0-9.]+)\}?\$", line)
        if len(cells) >= 4 and "&" in line:
            rows[line.split("&")[0].strip()] = [("B" if b else "") + v for b, v in cells[:4]]
    return rows


STATES = [("GRPO_LA0", 0)] + [(a, it) for a in ("GRPO_LA0", "GRPO_LA5") for it in range(1, 11)]


def build(judge: str) -> list[str]:
    proc, pers, paired = load(judge)
    # base rows of the two arms are the same shared Base
    for key, src, lcol, *_ in COLS:
        assert abs(level(proc, pers, "GRPO_LA0", 0, src, lcol)
                   - level(proc, pers, "GRPO_LA5", 0, src, lcol)) < 1e-12, key
    stars = {}
    for key, _, _, pcol, _, low in COLS:
        p = paired[paired.metric == pcol]
        assert sorted(p.iteration) == list(range(1, 11)), (judge, pcol)
        assert set(p.lower_better) == {low}, (judge, pcol, "lower-is-better flag disagrees")
        for _, r in p.iterrows():
            if r.p_holm < 0.05:
                stars[(key, ARMS["GRPO_LA0" if r.better == "K0" else "GRPO_LA5"], int(r.iteration))] = True
    # check iteration 10 against the printed Table 3 / Table 8 (exact row labels): the Base and
    # iteration-10 levels, and the vs-10 dz VALUE (since 2026-10-07 the printed stars are Holm
    # across the table's rows, not across iterations, so only the value is compared here; the
    # anchor table itself is checked by anchor_rows()).
    t3, dzp = table3_rows(*JUDGES[judge]), table3_dz(*JUDGES[judge])
    for key, src, lcol, pcol, lab, low in COLS:
        printed = t3[lab]
        got = [level(proc, pers, "GRPO_LA0", 0, src, lcol),
               level(proc, pers, "GRPO_LA0", 10, src, lcol),
               level(proc, pers, "GRPO_LA5", 10, src, lcol)]
        for g, pv in zip(got, (printed[0], printed[2], printed[3])):
            assert f"{g:.3f}" == pv.lstrip("B"), (judge, key, g, pv)
        p10 = paired[(paired.metric == pcol) & (paired.iteration == 10)]
        assert dz_cell(p10.iloc[0]).split("^")[0] == dzp[lab][1].split("^")[0], (judge, key, dzp[lab])
    # Lior, 2026-09-30: "bold each column with best score" -- the best printed value of each
    # column over all 21 states (lowest where lower is better); ties at three decimals all bold.
    best = {}
    for key, src, lcol, _, _, low in COLS:
        vals = [round(level(proc, pers, a, it, src, lcol), 3) for a, it in STATES]
        best[key] = min(vals) if low else max(vals)

    def row(arm, it, name):
        cells = []
        for key, src, lcol, *_ in COLS:
            x = level(proc, pers, arm, it, src, lcol)
            v = f"${x:.3f}$" if round(x, 3) != best[key] else f"$\\mathbf{{{x:.3f}}}$"
            if it and stars.get((key, ARMS[arm], it)):
                v += r"\rlap{$^{*}$}"
            cells.append(v)
        return f"{name} & " + " & ".join(cells) + r"\\"

    out = [row("GRPO_LA0", 0, "Base"), r"\midrule", r"\multicolumn{13}{@{}l}{\emph{$K{=}0$}}\\"]
    out += [row("GRPO_LA0", it, str(it)) for it in range(1, 11)]
    out += [r"\midrule", r"\multicolumn{13}{@{}l}{\emph{$K{=}5$}}\\"]
    out += [row("GRPO_LA5", it, str(it)) for it in range(1, 11)]
    n_star = {a: sum(1 for (k, aa, i) in stars if aa == a) for a in ("K0", "K5")}
    print(f"% {judge}: {len(stars)} starred cells (K=0 {n_star['K0']}, K=5 {n_star['K5']}); "
          f"iteration 10 matches {JUDGES[judge][1]}", file=sys.stderr)
    for k0 in sorted((k, i) for (k, a, i) in stars if a == "K0"):
        print(f"%   K=0 star: {k0}", file=sys.stderr)
    return out


# --- every code at the Base and iteration 10, both judges (Lior's read, 2026-10-05) ------------
# "Do we have tables for the per-utterance classification?" -- Tables 3 and 5 carry only the
# three codes the text discusses; the full mix was a stacked figure. Grouped as the coder's
# MI-adherent / MI-inconsistent sums are (``eda_analysis.process.MI_ADHERENT`` / ``MI_INCONSISTENT``).
CODE_GROUPS = [
    ("MI-consistent", [("OQ", "open question"), ("SR", "simple reflection"),
                     ("CR", "complex reflection"), ("AF", "affirmation"),
                     ("SEEK", "seeking collaboration")]),
    ("MI-inconsistent", [("PRA", "non-specific praise"), ("PERS", "persuasion"),
                         ("CONF", "confront")]),
    ("Neither", [("CQ", "closed question"), ("GI", "giving information"), ("OTH", "other")]),
]


def table3_dz(tex_name: str, label: str) -> dict[str, tuple[str, str]]:
    """The printed dz cells of each Table 3 / Table 8 row: label -> (vs K=0's best, vs K=0 at 10),
    e.g. ('-0.73^{***}', '-1.19^{***}')."""
    tex = (SECTIONS / tex_name).read_text(encoding="utf-8")
    body = tex[:tex.index(f"\\label{{{label}}}")]
    body = body[body.rindex("\\begin{tabular}"):]
    out = {}
    for line in body.splitlines():
        parts = [p.strip() for p in line.rstrip().rstrip("\\").split("&")]
        if len(parts) == 8:
            out[parts[0]] = (parts[5].strip("$"), parts[6].strip("$"))
    return out


def dz_cell(r) -> str:
    """K=5 − K=0 dz (the paired sheets store K=0 − K=5) with Holm stars, as Table 3 prints it;
    '---' where dz is undefined (both runs at zero in every conversation)."""
    if pd.isna(r.dz):
        return "---"
    stars = "".join("*" for a in (0.05, 0.01, 0.001) if r.p_holm < a)
    return f"{-r.dz:+.2f}" + (f"^{{{stars}}}" if stars else "")


def codes() -> list[str]:
    """Rows of the code-mix table: per code, Base / K=0 / K=5 at iteration 10 and the paired dz,
    training oracle then held-out judge. The praise, complex-reflection and persuasion cells are
    checked against Tables 3 and 6 as printed."""
    per_judge = {}
    for judge, (tex_name, label) in JUDGES.items():
        proc, pers, paired = load(judge)
        printed, printed_dz = table3_rows(tex_name, label), table3_dz(tex_name, label)
        cells = {}
        for _, group in CODE_GROUPS:
            for code, _ in group:
                col = f"th_{code}_rate"
                vals = [level(proc, pers, "GRPO_LA0", 0, "proc", col),
                        level(proc, pers, "GRPO_LA0", 10, "proc", col),
                        level(proc, pers, "GRPO_LA5", 10, "proc", col)]
                p = paired[(paired.metric == col) & (paired.iteration == 10)]
                assert len(p) == 1, (judge, col)
                dz = dz_cell(p.iloc[0])
                cells[code] = [f"${v:.3f}$" for v in vals] + [dz if dz == "---" else f"${dz}$"]
                lab = next((lab for _, _, lc, _, lab, _ in COLS if lc == col), None)
                if lab is not None:                      # in Table 3 / 8: must match exactly
                    pr = printed[lab]
                    assert [f"{v:.3f}" for v in vals] == [x.lstrip("B") for x in (pr[0], pr[2], pr[3])], (judge, col)
                    # value only: Table 3 stars Holm across its rows, this table across iterations
                    assert dz_cell(p.iloc[0]).split("^")[0] == printed_dz[lab][1].split("^")[0], (judge, col)
        per_judge[judge] = cells
    out = []
    for i, (gname, group) in enumerate(CODE_GROUPS):
        if i:
            out.append(r"\midrule")
        out.append(rf"\multicolumn{{9}}{{@{{}}l}}{{\emph{{{gname}}}}}\\")
        for code, name in group:
            row = [per_judge[j][code] for j in JUDGES]
            out.append(f"{name} & " + " & ".join(row[0] + row[1]) + r"\\")
    print("% codes: praise / complex reflection / persuasion match Tables 3 and 7", file=sys.stderr)
    return out


# --- the two-anchor endpoint tables (review round 4, Lior 2026-10-07) ---------------------------
# "Not sure about the tables of iteration 10 ... have also a table of best iterations (8 and 10)."
# Table 1 (the nine instruments, d_z under both judges) and Tables 3 / 8 (the twelve process
# measures, one judge each): the final K=5 policy against the K=0 run at its best checkpoint (chosen
# once, on the training oracle's Q1+Q2) and at its last iteration, plus the iterations at which the
# per-iteration K contrast is significant. d_z and stars: ``anchor_contrasts`` (Holm across the
# table's rows per judge and anchor); levels: ``levels_long`` / the process level sheets;
# iterations: ``significant_iterations`` (instruments) or the paired sheets (process).
INSTR = [("Q1Q2", r"Q1+Q2 (reward)"), ("Q1", r"\quad Q1"), ("Q2", r"\quad Q2"), ("WAI-SR", "WAI-SR"),
         ("CSQ-8", "CSQ-8"), ("MI-SAT", "MI-SAT"), ("MITI", "MITI"), ("PCT", "PCT (share)"),
         ("MICI", r"MICI (per turn, $\downarrow$)")]
PRIMARY, HELDOUT = "gpt-4o-mini", "claude-haiku-4-5"
#: cross-sheet tolerance: most sheets are saved at four decimals (levels_long and anchor_contrasts at six)
TOL = 6e-5
GROUPS = [("Therapist turns (share by dominant function)", ["pra", "cr", "pers", "adh", "inc"]),
          (r"The therapist's reply to the patient's \ldots", ["refl_ct", "pra_st", "pers_st", "refl_st"]),
          (r"Next patient utterance is change talk, after \ldots", ["ct_ct", "st_ct"]),
          ("The session (share of all patient utterances)", ["ct_prop"])]


def anchors() -> tuple[pd.DataFrame, int]:
    a = pd.read_excel(SHARED_BASE_XLSX, sheet_name="anchor_contrasts")
    best = sorted(set(int(i) for i in a.loc[a.anchor == "best_K0", "iter_K0"]))
    assert len(best) == 1, best
    assert set(a.iter_K5) == {10} and set(a.loc[a.anchor == "last", "iter_K0"]) == {10}
    return a, best[0]


def dz_anchor(r) -> str:
    """K=5 − K=0 dz with Holm stars (anchor_contrasts is already in the paper's sign)."""
    s = "".join("*" for t in (0.05, 0.01, 0.001) if r.p_holm < t)
    return f"${r.dz_K5_minus_K0:+.2f}" + (f"^{{{s}}}" if s else "") + "$"


def runs(its: list[int]) -> str:
    """[4, 6, 7, 8, 9, 10] -> '4, 6--10'."""
    its, out, i = sorted(its), [], 0
    while i < len(its):
        j = i
        while j + 1 < len(its) and its[j + 1] == its[j] + 1:
            j += 1
        out.append(str(its[i]) if j == i else (f"{its[i]}, {its[j]}" if j == i + 1 else f"{its[i]}--{its[j]}"))
        i = j + 1
    return ", ".join(out)


def iter_cell(k5: list[int], k0: list[int]) -> str:
    """Significant iterations, in K=5's favour unless marked ($K{=}0$)."""
    parts = ([runs(k5)] if k5 else []) + ([runs(k0) + r" ($\Kz$)"] if k0 else [])
    return "; ".join(parts) if parts else "---"


def _ints(s) -> list[int]:
    return [] if pd.isna(s) or str(s).strip() == "" else [int(float(x)) for x in str(s).split(",")]


def endpoint_rows() -> list[str]:
    """Table 1's rows: levels under the training oracle (two decimals, no bold), d_z of K=5 at 10
    against K=0 at its best checkpoint and at 10 under both judges, and the training oracle's
    significant iterations. Checks: the 'last' d_z equal k_contrast at iteration 10, and the
    anchor's paired means equal the level sheet (n = 96 on every instrument)."""
    a, best = anchors()
    a = a[a.family == "instruments"]
    lv = pd.read_excel(SHARED_BASE_XLSX, sheet_name="levels_long")
    kc = pd.read_excel(SHARED_BASE_XLSX, sheet_name="k_contrast")
    sig = pd.read_excel(SHARED_BASE_XLSX, sheet_name="significant_iterations")
    out = []
    for m, lab in INSTR:
        def L(arm, it, j=PRIMARY):
            r = lv[(lv.judge == j) & (lv.arm == arm) & (lv.metric == m) & (lv.iteration == it)]
            assert len(r) == 1, (j, arm, m, it)
            return float(r.iloc[0]["mean"])
        assert abs(L("GRPO_LA0", 0) - L("GRPO_LA5", 0)) < 1e-12, m
        vals = [L("GRPO_LA0", 0), L("GRPO_LA0", best), L("GRPO_LA0", 10), L("GRPO_LA5", 10)]
        cells = [f"${v:.2f}$" for v in vals]
        for j in (PRIMARY, HELDOUT):
            for anc, it0 in (("best_K0", best), ("last", 10)):
                r = a[(a.judge == j) & (a.anchor == anc) & (a.metric == m)]
                assert len(r) == 1 and int(r.iloc[0].n) == 96, (j, anc, m)
                r = r.iloc[0]
                assert abs(r.mean_K0 - L("GRPO_LA0", it0, j)) < TOL and abs(r.mean_K5 - L("GRPO_LA5", 10, j)) < TOL
                if anc == "last":
                    k = kc[(kc.judge == j) & (kc.metric == m) & (kc.iteration == 10)].iloc[0]
                    assert abs(r.dz_K5_minus_K0 - k.dz_K5_minus_K0) < TOL, (j, m)
                cells.append(dz_anchor(r))
        s = sig[(sig.judge == PRIMARY) & (sig.metric == m)].iloc[0]
        cells.append(iter_cell(_ints(s.iters_K5_better), _ints(s.iters_K0_better)))
        out.append(f"{lab} & " + " & ".join(cells) + r"\\")
    print(f"% Table 1: K=0's best checkpoint = iteration {best}; 'last' d_z match k_contrast", file=sys.stderr)
    return out


def anchor_rows(judge: str) -> list[str]:
    """Table 3 (training oracle) / Table 8 (held-out judge) rows: levels (three decimals; bold =
    each row's best of the four, the Base included, ties all bold), d_z of K=5 at 10 against K=0 at
    its best checkpoint and at 10 (stars: Holm across the twelve rows), and the iterations at which
    the per-iteration contrast is significant (Holm across iterations, as Tables 5 / 10 star them).
    Checks: the 'last' d_z equal the paired sheets' iteration-10 d_z."""
    a, best = anchors()
    a = a[(a.family == "process") & (a.judge == judge)]
    proc, pers, paired = load(judge)
    by_key = {c[0]: c for c in COLS}
    out = []
    for gi, (gname, keys) in enumerate(GROUPS):
        if gi:
            out.append(r"\midrule")
        out.append(rf"\multicolumn{{8}}{{@{{}}l}}{{\emph{{{gname}}}}}\\")
        for key in keys:
            _, src, lcol, pcol, lab, low = by_key[key]
            vals = [level(proc, pers, "GRPO_LA0", 0, src, lcol), level(proc, pers, "GRPO_LA0", best, src, lcol),
                    level(proc, pers, "GRPO_LA0", 10, src, lcol), level(proc, pers, "GRPO_LA5", 10, src, lcol)]
            r3 = [round(v, 3) for v in vals]
            b3 = min(r3) if low else max(r3)
            cells = [f"$\\mathbf{{{v:.3f}}}$" if round(v, 3) == b3 else f"${v:.3f}$" for v in vals]
            for anc in ("best_K0", "last"):
                r = a[(a.anchor == anc) & (a.metric == pcol)]
                assert len(r) == 1, (judge, anc, pcol)
                r = r.iloc[0]
                if anc == "last":
                    p10 = paired[(paired.metric == pcol) & (paired.iteration == 10)].iloc[0]
                    assert abs(r.dz_K5_minus_K0 + p10.dz) < TOL, (judge, pcol)
                cells.append(dz_anchor(r))
            p = paired[(paired.metric == pcol) & (paired.p_holm < 0.05)]
            cells.append(iter_cell([int(i) for i in p.loc[p.better == "K5", "iteration"]],
                                   [int(i) for i in p.loc[p.better == "K0", "iteration"]]))
            out.append(f"{lab} & " + " & ".join(cells) + r"\\")
    print(f"% {judge} process anchors: K=0's best = iteration {best}; 'last' d_z match the paired sheets",
          file=sys.stderr)
    return out


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if "--codes" in argv:
        print("\n".join(codes()))
        return 0
    if "--endpoint" in argv:
        print("\n".join(endpoint_rows()))
        return 0
    if "--anchors" in argv:
        for judge in JUDGES:
            print(f"% ---- {judge} ----")
            print("\n".join(anchor_rows(judge)))
        return 0
    for judge in JUDGES:
        print(f"% ---- {judge} ----")
        print("\n".join(build(judge)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
