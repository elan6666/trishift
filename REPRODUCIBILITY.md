# TriShift Reproducibility Guide

This guide describes the intended order for reproducing the repository outputs.
Large datasets, embeddings, trained models, paper figures, and local draft files are intentionally not tracked by git.

## 1. Clone and install

```bash
git clone https://github.com/elan6666/trishift.git
cd trishift
pip install -e .
```

Check that the core package imports:

```bash
python -c "from trishift import TriShift, TriShiftData; import trishift; print(trishift.__version__)"
```

For public benchmark preparation and baseline wrappers, use the baseline environment:

```bash
conda env create -f environment_baselines.yml
conda activate trishift-baselines
pip install -e .
```

Install the Torch/PyG stack required by GEARS according to your CUDA version.

## 2. Run the built-in mini demo

Before downloading the full datasets, verify the core TriShift code path:

```bash
python examples/adamson_mini/run_demo.py
```

This uses a small Adamson subset committed under `examples/adamson_mini/data`.
It trains and evaluates a tiny TriShift model and writes outputs to:

```text
artifacts/demo/adamson_mini
```

This demo is a smoke test only; it is not used for paper metrics.

## 3. Prepare full benchmark data

Download and prepare the public benchmark datasets:

```bash
python scripts/data/download_and_prepare_benchmark_data.py --datasets adamson dixit norman
```

The script uses `GEARS/PertData`, creates GEARS-native folders under `src/data/Data_GEARS`, and syncs processed h5ad files to:

```text
src/data/<dataset>/perturb_processed.h5ad
```

`src/data/` is ignored by git.

## 4. Prepare gene embeddings

Place gene embeddings under:

```text
src/data/Data_GeneEmbd
```

The default `configs/paths.yaml` expects:

| Config key | Expected local file | Source |
| --- | --- | --- |
| `emb_a` | `src/data/Data_GeneEmbd/ensem_emb_gpt3.5all_new.pickle` | scELMo library, file `Gene-GPT 3.5`: <https://sites.google.com/yale.edu/scelmolib> |
| `emb_b` | `src/data/Data_GeneEmbd/GenePT_gene_embedding_ada_text.pickle` | GenePT Zenodo record: <https://zenodo.org/records/10833191> |
| `emb_c` | `src/data/Data_GeneEmbd/GPT_3_5_gene_embeddings.pickle` | GenePT Zenodo record: <https://zenodo.org/records/10030426> |
| `emb_d` | `src/data/Data_GeneEmbd/GenePT_gene_protein_embedding_model_3_text.pickle` | Optional GenePT protein/text embedding used only if selected in custom configs |

If your files live elsewhere, edit `configs/paths.yaml`.

## 5. Train and evaluate TriShift

Run TriShift first, because downstream reference-conditioning and figure notebooks expect TriShift result folders.

```bash
python scripts/trishift/adamson/run_adamson.py
python scripts/trishift/dixit/run_dixit.py
python scripts/trishift/norman/run_norman.py
```

For a quick code-path validation on full data, add `--fast`.

Outputs are written under:

```text
artifacts/results/<dataset>
```

## 6. Run baseline methods

Run baselines after the full h5ad files and embeddings are available.
The usual order is GEARS and Scouter first, then the additional representation/foundation-model baselines.

GEARS:

```bash
python scripts/gears/adamson/run_gears_adamson.py
python scripts/gears/dixit/run_gears_dixit.py
python scripts/gears/norman/run_gears_norman.py
```

Scouter:

```bash
python scripts/scouter/adamson/run_scouter_adamson.py
python scripts/scouter/dixit/run_scouter_dixit.py
python scripts/scouter/norman/run_scouter_norman.py
```

biolord:

```bash
python scripts/biolord/adamson/run_biolord_adamson.py
python scripts/biolord/dixit/run_biolord_dixit.py
python scripts/biolord/norman/run_biolord_norman.py
```

GenePert:

```bash
python scripts/genepert/adamson/run_genepert_adamson.py
python scripts/genepert/dixit/run_genepert_dixit.py
python scripts/genepert/norman/run_genepert_norman.py
```

scGPT:

```bash
python scripts/scgpt/adamson/run_scgpt_adamson.py
python scripts/scgpt/dixit/run_scgpt_dixit.py
python scripts/scgpt/norman/run_scgpt_norman.py
```

For scGPT, set `scgpt_pretrained_root` in `configs/paths.yaml` to a local pretrained scGPT model directory before running.

## 7. Run Systema-style analyses

After TriShift and baseline predictions exist, run Systema-style baseline and reference-comparison analyses.

```bash
python scripts/systema/adamson/run_systema_baselines_adamson.py
python scripts/systema/dixit/run_systema_baselines_dixit.py
python scripts/systema/norman/run_systema_baselines_norman.py
```

Reference-conditioning comparison for Adamson:

```bash
python scripts/systema/adamson/run_systema_ref_compare_adamson.py
```

## 8. Generate paper figures

Run notebooks only after the result folders have been produced.
The recommended order is:

```text
notebooks/Fig1_MethodOverview.ipynb
notebooks/Fig2_MultiDatasetBenchmark.ipynb
notebooks/Fig3_ReferenceConditioning.ipynb
notebooks/Fig4_NormanGeneralization.ipynb
notebooks/Fig5_DistributionRecovery.ipynb
notebooks/FigS1_BenchmarkExtension.ipynb
notebooks/FigS2_AdditionalCases.ipynb
notebooks/FigS3_BiologyAndAblation.ipynb
notebooks/FigS4_CentroidAnalysis.ipynb
notebooks/FigS5_Robustness.ipynb
notebooks/FigS6_Stage1LatentClustering.ipynb
```

Figure outputs are written under:

```text
artifacts/paper_figures
```

## 9. Expected local-only directories

The following directories are intentionally local and ignored by git:

- `src/data/`
- `artifacts/`
- `external/`
- `docs/`
- `output/`

Do not use ignored directories as public repository entrypoints. Use the scripts and configs tracked in git.
