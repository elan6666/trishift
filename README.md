# TriShift: Tripartite Reference-Conditioned Shift Model

TriShift, short for `Tripartite Reference-Conditioned Shift Model`, is a single-cell perturbation response prediction project built around the `Tripartite Reference-Conditioned Shift Model` (`TriShift`, 中文全名：`三元参考条件化状态转移模型`) and its benchmark stack:

- the native `trishift` model (`Tripartite Reference-Conditioned Shift Model`, `TriShift`)
- evaluation wrappers for `scouter`
- evaluation wrappers for `GEARS`
- evaluation wrappers for `biolord`
- evaluation wrappers for `genepert`
- evaluation wrappers for `scgpt`
- Systema-style reference and metric baselines
- a shared metric stack used across models

The repository uses a `src/` layout and is installable as a Python package.

## Package Install

Editable install:

```bash
pip install -e .
```

After installation:

```bash
python -c "import trishift; print(trishift.__version__)"
```

The package source lives in:

- `src/trishift`

Key runtime config files:

- `configs/defaults.yaml`
- `configs/paths.yaml`

## Recommended Environment

The current validated local environment is the `scouter` environment.

- Python: `3.10.19`
- PyTorch: `2.5.1`
- CUDA runtime: `12.1`
- NumPy: `2.0.1`
- Pandas: `2.3.3`
- SciPy: `1.15.3`
- Scanpy: `1.11.5`
- AnnData: `0.11.4`
- scikit-learn: `1.7.2`
- `scouter-learn`: `0.1.10`
- `cell-gears`: `0.1.2`
- `POT`: `0.9.6.post1`

### Minimal Setup Flow

Create a clean environment first:

```bash
conda create -n trishift python=3.10
conda activate trishift
```

Install PyTorch first, matching your CUDA setup. The validated environment uses CUDA 12.1:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Install the package:

```bash
pip install -e .
```

Install evaluation backends as needed:

```bash
pip install scouter-learn cell-gears POT
```

For GEARS, you also need the PyG stack that matches your Torch/CUDA version:

- `pyg-lib`
- `torch-geometric`
- `torch-scatter`
- `torch-sparse`

Install those using the wheel index recommended by PyG for your exact Torch build.

## Data Download And Preparation

### 1. Download Gene Embeddings

Gene embedding download links are documented in:

- `src/data/Data_GeneEmbd/README.md`

Download the required files there and place them under:

- `src/data/Data_GeneEmbd`

### 2. Download and Prepare Benchmark Data

Prepare the public benchmark datasets with:

```bash
python scripts/data/download_and_prepare_benchmark_data.py --datasets adamson dixit norman
```

This TriShift entrypoint delegates raw dataset download to `GEARS/PertData`, prepares the standard simulation splits, and then synchronizes the processed `perturb_processed.h5ad` files to the paths expected by TriShift and the baseline wrappers.

The legacy command is still supported for backward compatibility:

```bash
python src/data/Data_GEARS/generating_data.py
```

The preparation script does two things:

1. builds the GEARS-native dataset folders under:
   - `src/data/Data_GEARS/adamson`
   - `src/data/Data_GEARS/dixit`
   - `src/data/Data_GEARS/norman`
   - `src/data/Data_GEARS/replogle_k562_essential`
   - `src/data/Data_GEARS/replogle_rpe1_essential`
2. copies each generated `perturb_processed.h5ad` into the standard outer data directories:
   - `src/data/adamson/perturb_processed.h5ad`
   - `src/data/dixit/perturb_processed.h5ad`
   - `src/data/norman/perturb_processed.h5ad`
   - `src/data/replogle_k562_essential/perturb_processed.h5ad`
   - `src/data/replogle_rpe1_essential/perturb_processed.h5ad`

This means:

- GEARS can read from `src/data/Data_GEARS`
- TriShift / Scouter / Systema can read from `src/data/<dataset>`

### 3. Path Resolution

By default, the repository now expects local data under `src/data`.

If needed, you can still override paths through:

- `configs/paths.yaml`

## Main Entrypoints

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

## Example Commands

TriShift:

```bash
python scripts/trishift/adamson/run_adamson.py
python scripts/trishift/norman/run_norman.py
```

Scouter:

```bash
python scripts/scouter/adamson/run_scouter_adamson.py
python scripts/scouter/norman/run_scouter_norman.py
```

GEARS:

```bash
python scripts/gears/adamson/run_gears_adamson.py
python scripts/gears/norman/run_gears_norman.py
```

BioLord:

```bash
python scripts/biolord/adamson/run_biolord_adamson.py
python scripts/biolord/norman/run_biolord_norman.py
```

Train through the shared dataset runner:

```bash
python scripts/trishift/train/run_dataset.py --dataset adamson
python scripts/trishift/train/run_dataset.py --dataset norman
```

## Notes

- legacy top-level `scripts/run_*` files, if present, should be treated as compatibility entrypoints rather than the primary maintained interface.
- model-specific implementations live under `scripts/trishift`, `scripts/scouter`, `scripts/gears`, `scripts/biolord`, `scripts/genepert`, `scripts/scgpt`, and `scripts/systema`.
- the project is already packaged through `pyproject.toml`; editable install is the expected workflow during development.
- experimental analysis and paper figures live under `notebooks`, including:
  - `notebooks/Fig1_MethodOverview.ipynb`
  - `notebooks/Fig2_MultiDatasetBenchmark.ipynb`
  - `notebooks/Fig3_ReferenceConditioning.ipynb`
  - `notebooks/Fig4_NormanGeneralization.ipynb`
  - `notebooks/Fig5_BiologyAndAblation.ipynb`
- paper drafts and supporting documentation live under `docs`, including:
  - `docs/trishift_paper_draft_cn.md`
  - `docs/eval_metrics_guide_cn.md`
  - `docs/bioinformatics_paper_outline_cn.md`
- run outputs are written under `artifacts`, with cached intermediates under `artifacts/cache` and evaluation results under `artifacts/results`.
