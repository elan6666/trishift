# TriShift

TriShift is a single-cell perturbation response prediction project with:

- the native `trishift` model
- evaluation wrappers for `scouter`
- evaluation wrappers for `GEARS`
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

### 2. Generate GEARS Data

Generate the GEARS datasets with:

```bash
python src/data/Data_GEARS/generating_data.py
```

This script now does two things:

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

TriShift:

- `scripts/run_adamson.py`
- `scripts/run_dixit.py`
- `scripts/run_norman.py`
- `scripts/run_k562.py`
- `scripts/run_rpe1.py`

Scouter evaluation:

- `scripts/run_scouter_eval_adamson.py`
- `scripts/run_scouter_eval_dixit.py`
- `scripts/run_scouter_eval_norman.py`
- `scripts/run_scouter_eval_k562.py`
- `scripts/run_scouter_eval_rpe1.py`

GEARS evaluation:

- `scripts/run_gears_eval_adamson.py`
- `scripts/run_gears_eval_dixit.py`
- `scripts/run_gears_eval_norman.py`
- `scripts/run_gears_eval_k562.py`
- `scripts/run_gears_eval_rpe1.py`

## Example Commands

TriShift:

```bash
python scripts/run_adamson.py
python scripts/run_norman.py
```

Scouter evaluation:

```bash
python scripts/run_scouter_eval_adamson.py
python scripts/run_scouter_eval_norman.py
```

GEARS evaluation:

```bash
python scripts/run_gears_eval_adamson.py
python scripts/run_gears_eval_norman.py
```

## Notes

- `scripts/run_*` files are compatibility entrypoints.
- model-specific implementations live under `scripts/trishift`, `scripts/scouter`, `scripts/gears`, and `scripts/systema`.
- the project is already packaged through `pyproject.toml`; editable install is the expected workflow during development.
