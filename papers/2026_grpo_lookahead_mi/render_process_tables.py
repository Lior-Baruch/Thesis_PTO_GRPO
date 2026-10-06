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
    """The iteration-10 rows of Table 3 / Table 5 as printed: label -> [Base, K0, K5] strings."""
    tex = (SECTIONS / tex_name).read_text(encoding="utf-8")
    body = tex[:tex.index(f"\\label{{{label}}}")]
    body = body[body.rindex("\\begin{tabular}"):]
    rows = {}
    for line in body.splitlines():
        cells = re.findall(r"\$(\\mathbf\{)?([0-9.]+)\}?\$", line)
        if len(cells) >= 3 and "&" in line:
            rows[line.split("&")[0].strip()] = [("B" if b else "") + v for b, v in cells[:3]]
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
    # check iteration 10 against the printed Table 3 / Table 7 (exact row labels): the levels, the
    # bold (Lior, 2026-10-05: each row's best of the Base, K=0 and K=5, ties at three decimals all
    # bold; until then the better run where the contrast is significant) and the dz cell + stars.
    t3, dz10 = table3_rows(*JUDGES[judge]), table3_dz(*JUDGES[judge])
    for key, src, lcol, pcol, lab, low in COLS:
        printed = t3[lab]
        got = [level(proc, pers, "GRPO_LA0", 0, src, lcol),
               level(proc, pers, "GRPO_LA0", 10, src, lcol),
               level(proc, pers, "GRPO_LA5", 10, src, lcol)]
        for g, pv in zip(got, printed):
            assert f"{g:.3f}" == pv.lstrip("B"), (judge, key, g, pv)
        r3 = [round(g, 3) for g in got]
        best3 = min(r3) if low else max(r3)
        assert [pv.startswith("B") for pv in printed] == [v == best3 for v in r3], (judge, key, printed)
        p10 = paired[(paired.metric == pcol) & (paired.iteration == 10)]
        assert dz_cell(p10.iloc[0]) == dz10[lab], (judge, key, dz10[lab])
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


def table3_dz(tex_name: str, label: str) -> dict[str, str]:
    """The printed dz cell of each Table 3 / Table 5 row: label -> e.g. '-1.19^{***}'."""
    tex = (SECTIONS / tex_name).read_text(encoding="utf-8")
    body = tex[:tex.index(f"\\label{{{label}}}")]
    body = body[body.rindex("\\begin{tabular}"):]
    out = {}
    for line in body.splitlines():
        parts = [p.strip() for p in line.rstrip("\\").split("&")]
        if len(parts) == 5:
            out[parts[0]] = parts[4].strip("$")
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
                if lab is not None:                      # in Table 3 / 6: must match exactly
                    assert [f"{v:.3f}" for v in vals] == [x.lstrip("B") for x in printed[lab]], (judge, col)
                    assert dz_cell(p.iloc[0]) == printed_dz[lab], (judge, col, printed_dz[lab])
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


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if "--codes" in argv:
        print("\n".join(codes()))
        return 0
    for judge in JUDGES:
        print(f"% ---- {judge} ----")
        print("\n".join(build(judge)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
