# Quick Reproducibility Path

This file is the shortest maintained path for validating a fresh TriShift checkout.
It does not attempt to rerun every baseline or regenerate every manuscript figure.

## 1. Clone and install

```bash
git clone https://github.com/elan6666/trishift.git
cd trishift
pip install -e .
```

Check the import:

```bash
python -c "from trishift import TriShift, TriShiftData; import trishift; print(trishift.__version__)"
```

## 2. Smoke test without external data

```bash
python examples/adamson_mini/run_demo.py
```

Expected output directory:

```text
artifacts/demo/adamson_mini
```

This validates the core training and evaluation path only. It is not a paper metric.

## 3. Prepare the three maintained benchmark datasets

Use the baseline environment when running the public benchmark data downloader because it depends on GEARS/PertData.

```bash
conda env create -f environment_baselines.yml
conda activate trishift-baselines
pip install -e .
python scripts/setup/bootstrap_external_baselines.py --only gears
python scripts/data/download_repro_inputs.py --items benchmark genept
```

The processed h5ad files should appear at:

```text
src/data/adamson/perturb_processed.h5ad
src/data/dixit/perturb_processed.h5ad
src/data/norman/perturb_processed.h5ad
```

## 4. Check gene embeddings

The unified downloader extracts the default GenePT embedding into:

```text
src/data/Data_GeneEmbd
```

For the default TriShift configs, the most important file is:

```text
src/data/Data_GeneEmbd/GenePT_gene_embedding_ada_text.pickle
```

Run:

```bash
python scripts/setup/check_repro_inputs.py --scope benchmark --strict
```

## 5. Run TriShift benchmark entrypoints

For a fast sanity check:

```bash
python scripts/trishift/adamson/run_adamson.py --fast
python scripts/trishift/dixit/run_dixit.py --fast
python scripts/trishift/norman/run_norman.py --fast
```

For the full run, omit `--fast`:

```bash
python scripts/trishift/adamson/run_adamson.py
python scripts/trishift/dixit/run_dixit.py
python scripts/trishift/norman/run_norman.py
```

Expected result files:

```text
artifacts/results/adamson/metrics.csv
artifacts/results/dixit/metrics.csv
artifacts/results/norman/metrics.csv
```

## 6. Optional baseline reproduction

Populate external baseline source trees:

```bash
python scripts/setup/bootstrap_external_baselines.py --only scgpt,gears,biolord,genepert
```

Prepare BioLORD inputs and scGPT checkpoint files through the same downloader:

```bash
pip install gdown
python scripts/data/download_repro_inputs.py --items biolord scgpt
```

Check baseline inputs:

```bash
python scripts/setup/check_repro_inputs.py --scope baselines
```

Then run the model-specific entrypoints under:

```text
scripts/gears/<dataset>/
scripts/biolord/<dataset>/
scripts/genepert/<dataset>/
scripts/scgpt/<dataset>/
```

## 7. Optional scGen PBMC case

```bash
python scripts/data/download_repro_inputs.py --items scgen protein genept
python scripts/trishift/scgen_pbmc_celltype/run_scgen_pbmc_celltype.py
```

If you only need the h5ad and not the protein-prior variants:

```bash
python scripts/data/download_repro_inputs.py --items scgen --skip-scgen-priors
```

## 8. Full manuscript workflow

For the complete benchmark, baseline, Systema, and notebook order, use:

```text
REPRODUCIBILITY.md
```

The manuscript source/PDF/supplement are maintained separately:

```text
https://github.com/elan6666/trishift-paper
```
