# Exp4_OpenStack — folder map

PTO vs GRPO on a **fully open model stack**: the oracle grader and the patient simulator are open
models served by a local vLLM endpoint, so a full run costs **$0 in API**. A side project off the
thesis; see [Exp3_PTO_GRPO](../Exp3_PTO_GRPO/) for the parent experiment.

> **The spec lives in [CLAUDE.md](CLAUDE.md)** — what Exp4 is, the algorithms, the module contract,
> the naming grammar, the data layout, the VRAM budget, and the gotchas. This file only maps the
> folder and says how to run things. Dated history goes in `history/` (created when there is any).

## Map

| | what it is |
|---|---|
| [code/](code/) | the trainers. `core/` is the shared layer; `grpo/` and `pto/` are the two methods; `tools/` holds the stand-alone utilities. `questionnaires.py` + `system_prompts_builder.py` are **verbatim copies from Exp3** and must not be edited — they are the measuring instrument and the task definition. |
| `data/` | **gitignored**; three Google Drive symlinks (see below). Schemas are documented in CLAUDE.md § "Data layout" — that is the only record of their shape. |
| [eda/](eda/) | the analysis: the `eda_analysis` package, one notebook per results family, and `tools/render_results.py`. |

## Quick start

### Train (Colab A100)

Open `code/grpo/train_grpo.ipynb` (or `code/pto/train_pto.ipynb`) from Drive in Colab and run
top-to-bottom. The cell order is a contract, not a style:

1. flat globals
2. runtime detect + auth
3. **`serve_roles()` — starts vLLM, before any torch import**
4. `import trl` **then** torch, model build
5. the visible orchestration loop

`EXPERIMENT_NAME` is **computed** from the config, never typed. Outputs land under
`data/runs/<EXPERIMENT_NAME>/`; conversations under `data/conversations/<EXPERIMENT_NAME>/`.

### Score + analyse (local)

```powershell
cd Exp4_OpenStack\eda
..\..\.venv\Scripts\python.exe -m eda_analysis._selfcheck --fast
..\..\.venv\Scripts\python.exe tools\render_results.py
```

Scoring runs from `eda/notebooks/scoring/Run_Eval.ipynb`. Arms are auto-discovered from disk —
there is no registry to edit when a new run appears.

### Smoke tests (local, no GPU needed for most)

```powershell
cd Exp4_OpenStack\code
..\..\.venv\Scripts\python.exe tools\smoke.py naming     # arm-name grammar round-trip
..\..\.venv\Scripts\python.exe tools\smoke.py all        # everything the local box can run
```

⚠ **`smoke.py` refuses over-budget VRAM plans before allocating.** On the local RTX 5070 Ti (12 GB)
an over-budget request **reboots the machine** — it does not raise `OutOfMemoryError`. That is why
batch size is a safety setting here, not a throughput knob.

### Oracle sanity — run this before spending a GPU-hour

```powershell
..\..\.venv\Scripts\python.exe tools\oracle_sanity.py --base-url http://localhost:8000/v1
```

An open-weights grader can honour the JSON schema perfectly and still return **degenerate** scores
(every item a 4, near-zero variance). That parses, writes valid parquet, and yields a grader that
cannot tell any two arms apart — and nothing downstream would flag it. The gate checks schema
validity and score variance against `tools/fixtures/sanity/transcripts.json`: 12 real Exp3
transcripts spanning Q1+Q2 **1.00 to 5.00** (SD 1.41), with their **gpt-4o-mini scores frozen from
the Exp3 score lake** as the reference. Regenerate the fixture with
`tools/fixtures/build_fixture.py` (reads Exp3 artifacts; costs nothing).

## Data & large artifacts

`data/` is gitignored. Its three subdirectories are **Google Drive directory symlinks**, so Colab
writes to mounted Drive and Drive Desktop surfaces the files straight inside the repo — the EDA
reads through the link unchanged.

Re-create them with Windows **Developer Mode** on, using `mklink` (**not** PowerShell
`New-Item -ItemType SymbolicLink` — WinPS 5.1 ignores Developer Mode and still demands admin):

```powershell
$D = "G:\My Drive\Thesis_PTO_GRPO\Exp4_OpenStack\data"
$R = "C:\Users\baruc\Desktop\Projects\Thesis_PTO_GRPO\Exp4_OpenStack\data"
cmd /c "mklink /D ""$R\runs""          ""$D\runs"""
cmd /c "mklink /D ""$R\conversations"" ""$D\conversations"""
cmd /c "mklink /D ""$R\eval_scores""   ""$D\eval_scores"""
```

To undo: delete the **link** (`Remove-Item "$R\runs"`) — the Drive data is untouched.

⚠ **"The dir reads as empty" does not mean "the run died."** The Drive mount can wedge on a single
folder (Exp3 saw a populated directory read as 0 files with an intermittent `WinError 1450` while
every file was present in Drive the whole time; a Drive restart fixed it). Check the cloud before
concluding an arm is unfinished.

Code is pushed to Drive for Colab **additively** — drag the `code` folder onto the Drive
`Exp4_OpenStack\` parent. Do **not** push `data/` (the symlink targets already live there) or
`eda/` (local-only). Never `robocopy /MIR` without asking first: it deletes on the destination.

## Relationship to Exp3

Exp4 is **additive** — nothing under `Exp3_PTO_GRPO/` is modified. It shares no data and no score
axis with Exp3 (different grader ⇒ different axis; compare within Exp4 only). It reuses Exp3's
rubrics and patient personas verbatim so that the **model stack is the only variable**, and it
fixes five Exp3 defects by construction — see CLAUDE.md § "Relationship to Exp3".
