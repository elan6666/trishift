# TriShift

TriShift is a single-cell perturbation response prediction project.

## Install (editable)

```
pip install -e .
```

## Run scripts

Example:

```
python scripts/run_adamson.py
```

### Script Layout (Updated)

- New official entrypoints are organized by model and dataset (for example `scripts/trishift/norman/run_norman.py`, `scripts/scouter/norman/run_scouter_norman.py`).
- Old flat `scripts/*.py` entrypoints are kept as compatibility wrappers for existing commands.
- Each dataset directory contains its local yaml config (for example `scripts/trishift/norman/config.yaml`).
- Transitional task-based wrappers/profiles (such as `scripts/trishift/train/`) are kept for compatibility during migration.

## Notes

- The package uses a src layout (`src/trishift`).
- Scripts read paths from `configs/paths.yaml` and settings from `configs/defaults.yaml`.
