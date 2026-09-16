"""Copy every figure this paper's .tex files reference from the tracked EDA results tree into
./figures/ (never symlink), applying a presentation crop where the EDA render carries an internal
title or footer that means nothing to a reader of the paper.

Figures are EDA-owned. Nothing here generates a figure: each entry points at an artifact rendered
by ``Exp3_PTO_GRPO/eda/tools/render_results.py``, or at a hand-authored method schematic under
``eda/results/schematics/``. Re-run after every render pass:

    & ..\\..\\.venv\\Scripts\\python.exe sync_figures.py           # copy (+ crop)
    & ..\\..\\.venv\\Scripts\\python.exe sync_figures.py --check   # report drift, copy nothing

**Figure scope policy.** This is the GRPO-ONLY paper — its subjects are the two GRPO arms, so
every results figure is a ``*_grpo`` artifact (recomputed/cropped to the 22 GRPO states); the
four-arm artifacts belonged to the 2x2 draft, retired 2026-09-04 to papers/archive/2026_pto_grpo_mi.
Figures read SCORES (levels incl. each arm's base), not K5-K0 deltas. The comparison axis is
ITERATIONS ONLY (decided 2026-08-27): no compute/budget or API-call figures.

**Four data figures are NOT copied by this script.** Body Figures 2 and 3
(``k_headline_q1q2_grpo.png``, ``overpraise_judgefree_grpo.png``) and appendix Figures 6 and 7
(``k_channel_forest_grpo_gpt-4o-mini.png``, ``tail_audit_grpo.png``) are drawn by
``render_paper_figures.py`` from the tracked TABLES behind the EDA renders (the renders themselves
are notebook-proportioned and illegible at ACL width); the saturation figure it also drew left the
paper on 2026-09-16. Re-run that script after a render pass, then this one.

**Crops (added 2026-09-02).** The EDA renders carry a suptitle naming the family and grader
("[EVAL] Judge saturation, GRPO arms only — …", "GRPO only — every instrument in LEVELS …") and
the schematic carries a footer that cross-references the PTO figure this paper does not contain.
``CROP`` removes exactly those bands, as fractions of the image, and nothing else; the caption in
the .tex carries the information instead. Data pixels are never touched.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import sys
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
RESULTS = REPO / "Exp3_PTO_GRPO" / "eda" / "results"
SCHEMATICS = RESULTS / "schematics"                 # hand-authored (no notebook, no judge level)
REWARD = RESULTS / "lookahead" / "reward" / "figures"
BEHAVIOUR = RESULTS / "lookahead" / "behaviour" / "figures"
MECHANISM = RESULTS / "lookahead" / "mechanism" / "figures"
MEASUREMENT = RESULTS / "measurement" / "validity" / "figures"
DEST = HERE / "figures"

# (source path, destination filename). Destination names are what the .tex references.
FIGURES: list[tuple[Path, str]] = [
    # sec:method — Figure 1 (method_grpo_group.png) is drawn by render_schematic.py (2026-09-14):
    # the EDA's portrait schematic (SCHEMATICS / "grpo_group_rollout.png") printed at ~4 pt in one
    # ACL column, so the paper carries a landscape figure* redraw of the same content instead.
    # sec:reward (k_headline_q1q2_grpo, since 2026-09-14) and sec:behaviour
    # (overpraise_judgefree_grpo) are drawn by render_paper_figures.py from the tracked tables --
    # see the module docstring. sec:measurement has had no figure since 2026-09-16.
    # --- appendix --------------------------------------------------------------------------------
    (REWARD / "k_levels_grid_grpo_gpt-4o-mini.png", "k_levels_grid_grpo_gpt-4o-mini.png"),
    (REWARD / "k_levels_grid_grpo_claude-haiku-4-5.png", "k_levels_grid_grpo_claude-haiku-4-5.png"),
    # The channel forest (k_channel_forest_grpo_gpt-4o-mini.png) and the rollout audit
    # (tail_audit_grpo.png) are drawn by render_paper_figures.py from behaviour.xlsx and
    # mechanism.xlsx since 2026-09-14 (paper sign convention, plain labels).
]

# destination name -> crop box as FRACTIONS of (width, height): (left, top, right, bottom).
# Each band removed is a title/footer line, checked by eye against the source render; the values
# sit in the whitespace between that line and the first plot element it would otherwise touch.
CROP: dict[str, tuple[float, float, float, float]] = {
    "k_levels_grid_grpo_gpt-4o-mini.png": (0.0, 0.035, 1.0, 1.0),
    "k_levels_grid_grpo_claude-haiku-4-5.png": (0.0, 0.035, 1.0, 1.0),
}


def _render(src: Path, name: str) -> bytes:
    """The bytes that belong in figures/<name>: the source, cropped if CROP says so."""
    if name not in CROP:
        return src.read_bytes()
    im = Image.open(src)
    w, h = im.size
    l, t, r, b = CROP[name]
    out = im.crop((round(l * w), round(t * h), round(r * w), round(b * h)))
    buf = io.BytesIO()
    out.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def _sha(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="report drift, copy nothing")
    a = ap.parse_args()
    DEST.mkdir(exist_ok=True)
    missing, drift, copied = [], [], []
    for src, name in FIGURES:
        dst = DEST / name
        if not src.exists():
            missing.append(str(src))
            continue
        want = _render(src, name)
        same = dst.exists() and _sha(want) == _sha(dst.read_bytes())
        if a.check:
            if not same:
                drift.append(name)
        elif not same:
            dst.write_bytes(want)
            copied.append(name)
    for m in missing:
        print("MISSING source:", m)
    if a.check:
        for d in drift:
            print("DRIFT:", d)
        print(f"{len(drift)} drifted, {len(missing)} missing")
    else:
        for c in copied:
            print("copied:", c)
        print(f"{len(copied)} copied, {len(FIGURES) - len(copied) - len(missing)} unchanged, {len(missing)} missing")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
