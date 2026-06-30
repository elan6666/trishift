# AGENTS.md

Project-level instructions for agents working in the TriShift code repository.

## Scope

- This repository owns model code, data utilities, evaluation scripts, notebooks,
  figure-generation provenance and server-side execution.
- Manuscript prose, captions, supplement text and compiled paper PDFs are owned
  by `/Users/elan/code/trishift-paper`. When a task touches manuscript text,
  follow that repository's `AGENTS.md`, where paper optimization starts from
  `byte-do`, uses `perturbation-prediction-writing` as the primary guide,
  applies `nature-polishing` only as an auxiliary layer, and ends with a
  `pre-submission-reviewer` gate for substantive edits.
- Do not stage or commit changes unless the user explicitly asks.

## Manuscript Touchpoints

- If code, notebooks or figure outputs require manuscript changes, update the
  paper repository in this order:
  1. Chinese manuscript;
  2. English manuscript;
  3. Supplement, only when definitions, provenance or extra diagnostics change.
- Treat the Chinese manuscript as the content source of truth for claim order,
  terminology, figure references, caveats and interpretation.
- Keep Chinese and English scientific content synchronized after every
  manuscript-relevant change.

## Figure Workflow

- Manuscript and supplement figures must be generated from notebooks under
  `notebooks/`.
- Do not use standalone Python render scripts as the final figure-generation
  entrypoint. Shared plotting helpers may live in `notebooks/_figure_helpers.py`.
- Active figure notebooks:
  - `notebooks/Fig2_ReferenceTransfer.ipynb`
  - `notebooks/Fig3_Ablation.ipynb`
  - `notebooks/Fig4_NormanGeneralization.ipynb`
  - `notebooks/Fig5_DistributionRecovery.ipynb`
  - `notebooks/FigS1_ReferenceTransferFull.ipynb`
  - `notebooks/FigS2_AdditionalCases.ipynb`
  - `notebooks/FigS3_DixitRobustness.ipynb`
  - `notebooks/FigS4_Module1LatentState.ipynb`
  - `notebooks/FigS5_NormanGeneralizationContext.ipynb`

## Local And Server Workflow

- Make code and notebook edits locally in `/Users/elan/code/trishift`.
- Sync edited repository files to `/data/yilangliu/trishift` before server
  notebook execution.
- Run figure notebooks on the server because full result folders and model
  payloads are server-side.
- Recover generated figure artifacts from the server back to the local code repo
  and then to the paper repo.
- Do not treat locally copied or manually composed images as final unless they
  were produced by server notebook execution.

## Server Safety

- Use the saved `server-access-yilangliu` workflow for remote work.
- Start remote commands from `/data/yilangliu/trishift` unless the user names
  another path.
- Never write server credentials into this repository, Byte OS files, shell
  scripts, notebooks, logs or chat output.
- Before overwriting remote files, preserve remote dirty state with a status
  snapshot and diff patch.

## Editing Boundaries

- Keep figure workflow changes focused on notebooks, notebook helpers, docs and
  provenance files.
- Do not rerun expensive training or baseline experiments unless the user
  explicitly asks or a required figure artifact is missing and the user approves.
- If a dataset, split, cell type, condition, panel or manuscript claim is
  ambiguous, ask before changing code, notebooks, manuscript text or figure
  artifacts.
- Work with existing dirty files; never revert user changes unless explicitly
  requested.

## Verification

- For code or notebook edits, run the narrowest relevant local checks plus:

```bash
git -c core.quotePath=false diff --check
git -c core.quotePath=false status --short
```

- For figure changes, record the notebook/server execution result and the
  recovered artifact paths.
- For manuscript-impacting figure changes, compile the affected paper PDFs in
  `/Users/elan/code/trishift-paper` before finishing.
