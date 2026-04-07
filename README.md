# TriShift: Tripartite Reference-Conditioned Shift Model

TriShift is a single-cell perturbation response prediction toolkit built around the `Tripartite Reference-Conditioned Shift Model` (`TriShift`). The repository contains the native TriShift implementation, shared evaluation code, benchmark wrappers for major baselines, and the notebooks used to generate the paper figures.

The project uses a `src/` layout and is installable as a Python package.

## For Users

### Install the core package

```bash
pip install -e .
```

After installation:

```bash
python -c "from trishift import TriShift, TriShiftData; import trishift; print(trishift.__version__)"
```

The core package source lives in:

- `src/trishift`

Key runtime config files:

- `configs/defaults.yaml`
- `configs/paths.yaml`

### Minimal custom-dataset tutorial

If you want to try TriShift on your own `AnnData`, start with:

- `notebooks/tutorial_custom_dataset.ipynb`

The tutorial shows a minimal workflow:

1. build a small `AnnData` with a `condition` column,
2. prepare a matching gene embedding table,
3. initialize `TriShiftData` and `TriShift`,
4. run a minimal train/evaluate loop,
5. export prediction payloads for downstream analysis.

### Public benchmark data

Prepare the public benchmark datasets with:

```bash
python scripts/data/download_and_prepare_benchmark_data.py --datasets adamson dixit norman
```

This entrypoint delegates raw data download to `GEARS/PertData`, prepares the standard simulation splits, and synchronizes `perturb_processed.h5ad` files to the paths expected by TriShift and the evaluation wrappers.

The legacy command is still supported for backward compatibility:

```bash
python src/data/Data_GEARS/generating_data.py
```

By default, the repository expects local data under `src/data`. You can still override locations through:

- `configs/paths.yaml`

## For Reproducibility

### Recommended environments

The lightest workflow is the core TriShift package:

```bash
pip install -e .
```

The benchmark stack mixes several external baselines with conflicting dependencies. To keep the main package usable, the repository separates:

- **Core TriShift dependencies** in `pyproject.toml`
- **Baseline-oriented environment setup** in `environment_baselines.yml`

Create the baseline environment with:

```bash
conda env create -f environment_baselines.yml
conda activate trishift-baselines
```

`environment_baselines.yml` covers the common stack used by `scouter`, `GEARS`, and shared evaluation tools. `GEARS` still requires a Torch/PyG installation matched to your local CUDA runtime; follow the comments in that file for the final install step.

### Data download and preprocessing

Gene embedding download links are documented in:

- `src/data/Data_GeneEmbd/README.md`

Download the required embedding files there and place them under:

- `src/data/Data_GeneEmbd`

The benchmark preparation script builds the GEARS-native dataset folders under:

- `src/data/Data_GEARS/adamson`
- `src/data/Data_GEARS/dixit`
- `src/data/Data_GEARS/norman`
- `src/data/Data_GEARS/replogle_k562_essential`
- `src/data/Data_GEARS/replogle_rpe1_essential`

It also copies each generated `perturb_processed.h5ad` into the standard outer data directories:

- `src/data/adamson/perturb_processed.h5ad`
- `src/data/dixit/perturb_processed.h5ad`
- `src/data/norman/perturb_processed.h5ad`
- `src/data/replogle_k562_essential/perturb_processed.h5ad`
- `src/data/replogle_rpe1_essential/perturb_processed.h5ad`

This keeps the repository consistent across:

- GEARS, which reads from `src/data/Data_GEARS`
- TriShift, Scouter, and Systema-style evaluation, which read from `src/data/<dataset>`

### Training and evaluation entrypoints

Recommended dataset entrypoints are organized by model and dataset under `scripts/<model>/<dataset>`.

TriShift:

- `scripts/trishift/adamson/run_adamson.py`
- `scripts/trishift/dixit/run_dixit.py`
- `scripts/trishift/norman/run_norman.py`
- `scripts/trishift/replogle_k562_essential/run_replogle_k562_essential.py`
- `scripts/trishift/replogle_rpe1_essential/run_replogle_rpe1_essential.py`

Scouter:

- `scripts/scouter/adamson/run_scouter_adamson.py`
- `scripts/scouter/dixit/run_scouter_dixit.py`
- `scripts/scouter/norman/run_scouter_norman.py`
- `scripts/scouter/replogle_k562_essential/run_scouter_k562.py`
- `scripts/scouter/replogle_rpe1_essential/run_scouter_rpe1.py`

GEARS:

- `scripts/gears/adamson/run_gears_adamson.py`
- `scripts/gears/dixit/run_gears_dixit.py`
- `scripts/gears/norman/run_gears_norman.py`
- `scripts/gears/replogle_k562_essential/run_gears_k562.py`
- `scripts/gears/replogle_rpe1_essential/run_gears_rpe1.py`

Additional baselines:

- `scripts/biolord/<dataset>/run_biolord_*.py`
- `scripts/genepert/<dataset>/run_genepert_*.py`
- `scripts/scgpt/<dataset>/run_scgpt_*.py`
- `scripts/systema/<dataset>/run_systema_*.py`

Shared training core:

- `scripts/trishift/_core/run_dataset_core.py`
- `scripts/trishift/train/run_dataset.py`

### Figure generation

The paper figures are generated from the notebooks under `notebooks/`:

- `Fig1_MethodOverview.ipynb` -> Fig. 1
- `Fig2_MultiDatasetBenchmark.ipynb` -> Fig. 2
- `Fig3_ReferenceConditioning.ipynb` -> Fig. 3
- `Fig4_NormanGeneralization.ipynb` -> Fig. 4
- `Fig5_BiologyAndAblation.ipynb` -> Fig. 5
- `FigS3_CentroidAnalysis.ipynb` -> Fig. S3
- `FigS4_Robustness.ipynb` -> Fig. S4
- `FigS7_Stage1LatentClustering.ipynb` -> Fig. S7

Primary outputs are written under:

- `artifacts/results`
- `artifacts/paper_figures`

### Tests

Tests now live in the repository root:

- `tests/`

Run the full test suite with:

```bash
pytest -q
```

Run a smaller smoke subset with:

```bash
pytest tests/test_data.py tests/test_eval.py -q
```

### Notes

- Legacy top-level `scripts/run_*` files, if present, should be treated as compatibility entrypoints rather than the primary maintained interface.
- Paper drafts and supporting documentation live under `docs`.
- Large local outputs, datasets, and external baseline clones are intentionally ignored by git.
