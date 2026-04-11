# Adamson Mini Demo

This directory contains a tiny Adamson-derived smoke-test dataset that can be committed to GitHub.
It is intended to verify that the TriShift training and evaluation code path runs end to end after cloning the repository.
It is not intended to reproduce the paper metrics.

## Files

- `data/adamson_mini.h5ad`: 320 cells by 160 genes, sampled from the public Adamson dataset.
- `data/genept_mini.pickle`: GenePT vectors for the perturbation genes present in the mini dataset.
- `config.yaml`: Small CPU-friendly TriShift config that points to the mini data.
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
