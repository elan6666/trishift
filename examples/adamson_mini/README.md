# Adamson Mini Demo

This directory contains a local Adamson-derived 10% subset for quicker TriShift validation.
It is intended to verify that the TriShift training and evaluation code path runs end to end with Adamson-like settings.
It is not intended to reproduce the paper metrics.

## Files

- `data/adamson_mini.h5ad`: a 10% cell subset of the public Adamson dataset.
- `config.yaml`: Adamson-like TriShift config with `1` split and `20` epochs.
- `run_demo.py`: One-command training and evaluation entrypoint.

## Run

Install the core package from the repository root:

```bash
pip install -e .
```

Then run:

```bash
python examples/adamson_mini/run_demo.py
```

The script writes outputs to:

```text
artifacts/demo/adamson_mini
```

`artifacts/` is ignored by git, so demo outputs will not be committed.
