# Looking Ahead in Goal-Oriented Dialogue: Comparing Preference-Tree and Group-Relative Optimization of Small Language Models for Motivational Interviewing

Master's thesis (Lior Baruch, Reichman University). We train small "therapist" LLMs
to conduct **Motivational Interviewing (MI)** against simulated patients, using a larger
**oracle** LLM that grades validated MI questionnaires as the reward signal. The work
extends our ICLR 2025 workshop paper on Preference-Tree Optimization (PTO).

## What it studies

Three controlled comparisons (all live in **Exp3**):

1. **Look-ahead depth** `K ∈ {0, 5}` — does anticipating future turns help, and by how much?
2. **PTO vs GRPO** under matched look-ahead `K` and minimum-conversation-length filter (MCL)
   — can iterative GRPO compete with PTO?
3. **Oracle questionnaire** (Q1+Q2 vs WAI-SR vs CSQ-8 vs MI-SAT/MITI) — held for later.

- **PTO** = the framework: grow a preference tree → `K`-turn look-ahead + oracle → τ-filtered
  (chosen, rejected) preference pairs → DPO update. The default `greedy` mode grows one trunk
  by appending the best-of-`M` completion at each therapist turn, so the choice feeds the next
  branch point (true PTO); the `independent` mode branches a pre-recorded conversation.
- **GRPO** = current policy simulates conversations → per-turn prompts (MCL filter) →
  GRPO update with optional `K`-turn look-ahead. The same conversations double as the eval set.

The held-out **evaluation** scores each conversation on a battery of validated MI questionnaires.
Because the warmth/satisfaction rubrics turned out highly collinear (one latent factor), the battery
now also includes **further metrics** — patient change-talk, MI-inconsistent behavior, and objective
MITI technique ratios (reflection-to-question, %complex-reflection) — so a uniform "everything went up"
can be distinguished from genuine multi-skill MI improvement.

## Repository layout

| Dir | Status | Therapist | Patient + oracle |
|---|---|---|---|
| [Exp1_ICLR2025/](Exp1_ICLR2025/) | Frozen — published | Llama-2-7B | GPT-3.5 |
| [Exp2_PTO/](Exp2_PTO/) | Complete — reference baseline | Llama-3.2-1B (4-bit NF4) | gpt-4o-mini |
| [Exp3_PTO_GRPO/](Exp3_PTO_GRPO/) | **Active** | Llama-3.2-1B (bf16) | gpt-4o-mini |

Each experiment directory is **self-contained**: its own `code/`, `eda/`, `data/`, and local
`system_prompts_builder.py` + `questionnaires.py` (versions deliberately diverge across
experiments). In Exp3 both helpers live once at `code/` root and the EDA package imports them.

Generated data (`data/`) is **not** tracked in git — see "Data & large artifacts" below.

### Write-ups

Two root-level directories hold everything written *for a human* rather than for the machine.
Both sit at the root rather than inside an `Exp*/` because they span experiments, and both only
ever *read* the generated artifacts under `Exp3_PTO_GRPO/eda/results/` — nothing in `code/` or
`eda/` imports them.

| Dir | What | Index |
|---|---|---|
| [`papers/`](papers/) | paper drafts, one subfolder per paper | [papers/README.md](papers/README.md) |
| [`meetings/`](meetings/) | supervisor decks + emails, one folder per date, plus the `build/` generators that produce them | [meetings/README.md](meetings/README.md) |

| Paper | Covers | Domain | Status |
|---|---|---|---|
| [`2025_iclr_pto_lookahead/`](papers/2025_iclr_pto_lookahead/) | PTO + look-ahead, as introduced | Exp1 | published, frozen |
| [`2026_grpo_lookahead_mi/`](papers/2026_grpo_lookahead_mi/) | *GRPO with Look-Ahead in Motivational Interviewing* — does scoring the continuation help a **group-relative** optimizer, and what does it change about the policy's behaviour? PTO cited as the lever's origin, never data | Exp3, the two GRPO arms, both graders | **THE submission — the single live paper, ARR October 2026** (submission 2026-10-12; feeds NAACL 2027 + COLING 2027; ACL format, iterations-only) |
| [`archive/2026_pto_grpo_mi/`](papers/archive/2026_pto_grpo_mi/) | *Same Lever, Different Optimizer* — the reward-horizon $\times$ optimizer interaction: the optimizer ranking flips with $K$ (iterations-only axis) | Exp3, all four arms, both graders | retired 2026-09-04 — Lior chose the GRPO-with-look-ahead paper as the single submission; complete draft, kept as a record |
| [`archive/2026_grpo_lookahead_mi/`](papers/archive/2026_grpo_lookahead_mi/) | *(same title — the ICLR-formatted version)* | Exp3, the two GRPO arms, both graders | retired 2026-08-27 when the ICLR plan was dropped; revived the same day as the live ACL/ARR draft above |
| [`archive/2026_lookahead_pto_grpo/`](papers/archive/2026_lookahead_pto_grpo/) | *Same Lever, Different Optimizer* — does $K$-turn look-ahead help, and does the answer depend on the optimizer? | Exp3, all four arms, both graders | retired 2026-08-25 — drafted against the right-censored GRPO K=5 arm, which has since finished at iteration 10 |
| [`archive/2026_clpsych_mi_reward_hacking/`](papers/archive/2026_clpsych_mi_reward_hacking/) | *Affirmation Without Inquiry* — what an LLM judge actually teaches, and how much of the gain a held-out judge credits | Exp3, K=0 only | retired 2026-08-18 — absorbed by *Same Lever*'s §6 |
| [`archive/2026_lookahead_hack_substitution/`](papers/archive/2026_lookahead_hack_substitution/) | *The Hack Moves* — trajectory-level reward redirects rather than reduces reward hacking | Exp3, K=5, PTO only | retired 2026-08-18 — absorbed by *Same Lever*'s §6 |

Every paper carries a **`NUMBERS.md` ledger** mapping each quantitative claim to the exact artifact
path it came from — so when the EDA is re-rendered and a number moves, you can find every sentence
that has to change. See [papers/README.md](papers/README.md).

### Where the docs live

| | |
|---|---|
| this file | setup, the API-key resolution order, and the data/artifact policy — including the Colab ↔ local sync mechanics (Drive symlinks, code push) |
| [CLAUDE.md](CLAUDE.md) | cross-experiment map + the full method/spec context for Exp3 |
| [STATUS.md](STATUS.md) | run status, headline numbers, cost constraint, next step |
| `Exp{1,2}_*/CLAUDE.md` | per-experiment context for the two frozen experiments |
| [Exp3_PTO_GRPO/eda/README.md](Exp3_PTO_GRPO/eda/README.md) | the EDA how-to: FAMILY/JUDGE knobs, package map, re-render |
| [Exp3_PTO_GRPO/code/README.md](Exp3_PTO_GRPO/code/README.md) | what each trainer module does |
| [Exp3_PTO_GRPO/history/](Exp3_PTO_GRPO/history/CHANGELOG.md) | the only dated history |

Docs are split by **rate of change**: CLAUDE.md describes how things are, STATUS.md is rewritten in
place each week, and anything dated goes to `history/`. CLAUDE.md's "Doc map" names the single owner
of every fact.

## Setup

Python 3.13.

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows (PowerShell: .venv\Scripts\Activate.ps1)

# Install torch from the CUDA wheel index matching your driver, then the rest:
pip install torch --index-url https://download.pytorch.org/whl/cu128   # cu121/cu118 for older drivers; omit for CPU
pip install -r requirements.txt
```

`requirements.txt` is generated by [gen_requirements.py](gen_requirements.py) and pins the
rest of the stack (transformers / trl / peft / accelerate, openai, the scientific stack).

### API keys

The patient simulator and oracle call the OpenAI API; the therapist base model is pulled from
Hugging Face. Keys are resolved at runtime — **Colab `userdata` secrets → environment variables →
local key files** — by `init_openai_client` / `authenticate` in
[Exp3_PTO_GRPO/code/_shared/runtime.py](Exp3_PTO_GRPO/code/_shared/runtime.py). Per secret:

| Secret | Colab | Local |
|---|---|---|
| OpenAI | `userdata["OPENAI_API_KEY"]` → env → file | env (`OPENAI_API_KEY`) → file |
| HF token | `userdata["huggingface"]` → env → file | env (`HF_TOKEN`/`HUGGINGFACE_TOKEN`) → file |
| W&B | `userdata["wandb"]` | env `WANDB_API_KEY` |

So provide them either way:

- Environment: `OPENAI_API_KEY`, and `HF_TOKEN` (or `HUGGINGFACE_TOKEN`), **or**
- Files in the experiment directory you're running: `openai_key.txt` and `HF_key.txt`.

The **HF token is used locally too** — Llama-3.2-1B is gated. On Colab all three secrets come from
**Colab Secrets** (`OPENAI_API_KEY`, `huggingface`, `wandb`), never the `.txt` files; the resolution
table above is the trainers' path.

These key files are git-ignored and must never be committed.

## Running

- **Training** (GPU; intended for Colab) — open the iterative trainer for a method, e.g.
  [Exp3_PTO_GRPO/code/GRPO_Exp3/train_GRPO_Iterative.ipynb](Exp3_PTO_GRPO/code/GRPO_Exp3/train_GRPO_Iterative.ipynb)
  or [.../PTO_Exp3/train_PTO_Iterative.ipynb](Exp3_PTO_GRPO/code/PTO_Exp3/train_PTO_Iterative.ipynb).
  Each notebook shows the per-iteration orchestration loop; shared helpers live in `code/_shared/`.
- **Evaluation & EDA** (local) — `Exp3_PTO_GRPO/eda/notebooks/scoring/Run_Eval.ipynb` runs the oracle scoring
  pipeline; the analysis lives in the `eda/eda_analysis/` package + fifteen family notebooks organised
  by research question (`eda/notebooks/{arms,lookahead,method,compute,measurement}/<sub>.ipynb`; the
  notebook path == its results family). Each notebook's cell 1 is
  `eda_analysis.EdaConfig(family="<top>/<sub>", judge=os.environ.get("EDA_JUDGE", ""))` →
  `notebook_setup(cfg)`; figures save as PNG and tables as Markdown + Excel (+ JSON ledgers) into
  `eda/results/<top>/<sub>/{figures,tables}/[<judge>/]` (`arms/*` per grader; the contrast tops carry
  both graders side by side). Regenerate everything with `python eda/tools/render_results.py`
  (`--top lookahead`, `--family arms/outcomes --judge <tag>` for a single unit). Full guide:
  [Exp3_PTO_GRPO/eda/README.md](Exp3_PTO_GRPO/eda/README.md).

## Data & large artifacts (not in git)

This repository tracks **code, notebooks, and docs only**. The following are intentionally
excluded by [.gitignore](.gitignore) because they are large and/or regenerable:

| Path | What it is | How it comes back |
|---|---|---|
| `**/data/` | Generated conversations, preference trees, oracle + second-judge eval scores (multiple GB) | Regenerated by running the pipelines (see below) |
| `.venv/` | Python virtual environment | `pip install -r requirements.txt` |
| `**/.emb_cache/`, `**/emb_cache_words/` | Sentence-embedding caches (Exp3 preference probe + archived Exp2) | Recomputed by the preference notebook |
| `**/eda/.eda_cache/` | Exp3 EDA parquet cache (`scores_long` / `behavior_by_iter`; content-keyed) | Rebuilt on next EDA run; `eda_analysis.reset_cache()` to clear |
| `HF_key.txt`, `openai_key.txt` | API credentials | Provide your own (see [API keys](#api-keys)) |

How it is regenerated:

- **Exp3 self-loop** — `GRPO_Exp3` and `PTO_Exp3` regenerate their training data each
  iteration from the current policy (current policy simulates conversations → per-turn
  prompts/branches → update). Those same conversations double as the evaluation set, so no
  separate generate-eval step is needed for trained iterations.
- **No cross-experiment data** — Exp3 is a fresh experiment that regenerates everything it needs from
  scratch; it shares no data with Exp2. (Exp2's own data lives under `Exp2_PTO/` and is regenerated by
  that experiment's pipelines.)
- **Evaluation** — `Exp3_PTO_GRPO/eda/notebooks/scoring/Run_Eval.ipynb` runs the oracle scoring pipeline to
  (re)produce the per-questionnaire eval scores into the score lake,
  `Exp3_PTO_GRPO/data/eval_scores/judge=<tag>/rep=<r>/`.

⚠ **The score lake is the only copy of the paid oracle + judge calls, and re-scoring it is not
affordable** — the spend to date and the cost constraint live in [STATUS.md](STATUS.md), which owns
that number. That is why it sits on a Google Drive symlink rather than local-only, and why
`data/eval_scores/_parquet/` matters: it folds the lake into a 31-file form that is cheap to copy
somewhere else again. Sync mechanics below.

⚠ **The `README.md` files inside the run directories are auto-generated model cards** — PEFT/TRL
writes one beside every saved adapter, so they appear at
`data/{grpo,pto}_Exp3/runs/**/iteration_*/adapter/README.md`,
`.../iteration_*/training/README.md` and `.../iteration_*/training/checkpoint-*/README.md`, in both
methods' runs. **Do NOT delete them or treat them as project docs.**

If you need to share the raw datasets or trained adapters, host them externally
(e.g. Hugging Face Hub, Zenodo, or Google Drive) rather than committing them here.

### Sync: Colab ↔ local

Training runs on Colab and writes into mounted Drive; the EDA runs locally and reads the same bytes.
No rclone is involved.

**Results pull — Google Drive Desktop.** `data/eval_scores`, `data/grpo_Exp3` and `data/pto_Exp3`
are all **directory symlinks** into `G:\My Drive\Thesis_PTO_GRPO\Exp3_PTO_GRPO\data\<name>`. Colab
writes to mounted Drive → Drive Desktop (kept in **streaming** mode, low disk) surfaces it locally →
the files appear straight inside the repo, and the EDA reads through the link unchanged (every read
goes via `WORKSPACE_ROOT/data/...`). The EDA only opens `conversations/` plus the score lake's
CSV/parquet, so streaming downloads just those on access; the big artifacts (`runs/`, adapters,
`*.safetensors`) are never read locally and also live on the HF Hub + W&B.

Re-create the links with Windows **Developer Mode** on, using `mklink` — **not** PowerShell
`New-Item -ItemType SymbolicLink`, which under WinPS 5.1 ignores Developer Mode and still demands
admin:

```powershell
$D = "G:\My Drive\Thesis_PTO_GRPO\Exp3_PTO_GRPO\data"
$R = "C:\Users\baruc\Desktop\Projects\Thesis_PTO_GRPO\Exp3_PTO_GRPO\data"
cmd /c "mklink /D ""$R\eval_scores"" ""$D\eval_scores"""
cmd /c "mklink /D ""$R\grpo_Exp3""   ""$D\grpo_Exp3"""
cmd /c "mklink /D ""$R\pto_Exp3""    ""$D\pto_Exp3"""
```

To undo, delete the **link** (`Remove-Item "$R\grpo_Exp3"`) — the Drive data is untouched.

**Code push (local → Drive, `code/` only) is manual and additive.** The whole `code/` tree lives at
`G:\My Drive\Thesis_PTO_GRPO\Exp3_PTO_GRPO\code\` — that is all Colab needs; open a
`train_*_Iterative.ipynb` from there. Do **not** push `data/` (the symlink targets already live in
Drive) or `eda/` (local-only). After editing code locally, push the update by **dragging the `code`
folder** onto the Drive `Exp3_PTO_GRPO\` parent — a merge that adds and overwrites but **never
deletes**. This is the default. Let Drive Desktop finish syncing (tray ✓) before running the Colab
cell.

⚠ **Never run `robocopy /MIR`, or any other delete-extras mirror, without Lior's explicit
go-ahead.** An exact mirror is the only thing that also *removes* files renamed or deleted
locally — which is precisely why it is destructive on the destination:

```powershell
# DESTRUCTIVE on the destination. Requires explicit go-ahead — see the warning above.
robocopy "C:\Users\baruc\Desktop\Projects\Thesis_PTO_GRPO\Exp3_PTO_GRPO\code" `
         "G:\My Drive\Thesis_PTO_GRPO\Exp3_PTO_GRPO\code" /MIR /XD __pycache__
```

rclone has the same asymmetry: `rclone sync A B` mirrors (deletes extras in B) — use `copy` for
additive and `check` for a dry-run diff.

## Hardware

Developed on Windows, RTX 5070 Ti (12 GB VRAM), CUDA 12.8, torch 2.11.0+cu128. Training is
intended for Colab GPUs; EDA and evaluation run locally.

## Citation

Lior Baruch, Reichman University — master's thesis (in progress), building on our ICLR 2025
workshop paper. _BibTeX to be added once finalized._

## License

Released under the MIT License — see [LICENSE](LICENSE).
