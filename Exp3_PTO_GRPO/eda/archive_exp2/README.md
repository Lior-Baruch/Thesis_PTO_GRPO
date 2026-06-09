# archive_exp2 — frozen Exp2-era EDA

These notebooks were built for **Exp2** (per-oracle `L0_/L5_` DPO arms, a single shared `Base`,
independent-group stats). They are kept here **frozen and runnable** for reference; do **not**
extend them. The live, Exp3-first EDA is in `eda/` (`exp3/` package + the `00/01/02/A` notebooks).

## Contents
- `Conv_EDA.ipynb` — Exp2 cross-model outcome metrics + stats battery.
- `Partial_Conv_Oracle_EDA.ipynb` — proxy-reliability diagnostic on Exp2 `Base` + `L5_Q1Q2_V*`.
- `pref_emb/preference_analysis.ipynb` — Exp2 preference-tree embedding analysis.
- `lib/` — a **frozen snapshot** of `eda/lib/` as of the archive date, so these notebooks keep
  running regardless of how the live `eda/lib/` evolves.

## How they resolve imports
Each notebook does a bare `from lib import …`, which binds to the `lib/` in the kernel's working
directory. `Conv_EDA` / `Partial_Conv_Oracle_EDA` sit directly beside the frozen `lib/`, so they
import it as-is. `pref_emb/preference_analysis.ipynb` is one level deeper, so its first code cell was
given a one-line `sys.path.insert(0, os.path.abspath(".."))` to reach the frozen `lib/`.

## Caveats
- These read **Exp2** data under `data/pto_Exp2/`. `pref_emb` additionally references a **legacy**
  path (`data/pto/pref_trees`, pre-rename) — it is preserved verbatim and may need the old data
  layout to run end-to-end. It was archived for provenance, not guaranteed to execute on current data.
- ⚠ The Exp2-era patient-characteristic join (`lib/data.add_patient_characteristics(patient_id=
  file_index)`) is **wrong for Exp3 iterative runs** (conversations are saved under a per-iteration
  *shuffled* index). The live `exp3/personas.py` fixes this. Don't use these notebooks on Exp3 data.

## Exp3 successor for the preference analysis
The `pref_emb` thread is carried forward (not dropped) by `eda/exp3/pref.py` + the preference section
of `eda/A_Algo_DeepDive.ipynb` (chosen-vs-rejected score margins, completion-embedding geometry, and
drift across iterations on the PTO_Exp3 `pref_pairs`).
