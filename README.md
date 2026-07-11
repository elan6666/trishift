<div align="center">
  <img src="assets/branding/trishift-avatar.png" alt="TriShift avatar" width="152" />

  <h1>TriShift</h1>

  <p><strong>让 AI 从状态匹配的参考细胞出发，预测未见扰动下的单细胞响应。</strong></p>
  <p><em>A reference-conditioned framework for predicting single-cell perturbation responses beyond observed perturbation space.</em></p>

  <p>
    <a href="https://github.com/elan6666/trishift/stargazers"><img src="https://img.shields.io/github/stars/elan6666/trishift?style=flat-square&color=5B8FF9" alt="GitHub stars" /></a>
    <a href="https://github.com/elan6666/trishift/network/members"><img src="https://img.shields.io/github/forks/elan6666/trishift?style=flat-square&color=8B5CF6" alt="GitHub forks" /></a>
    <a href="pyproject.toml"><img src="https://img.shields.io/badge/python-%3E%3D3.10-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python >= 3.10" /></a>
    <a href="https://github.com/elan6666/trishift-paper"><img src="https://img.shields.io/badge/paper-source-0F766E?style=flat-square" alt="Paper source" /></a>
  </p>

  <p><a href="#for-users">English</a> · <a href="#为什么选择-trishift">中文</a></p>
</div>

---

## 📰 News

- **[2026-07-11]** The public repository now uses the manuscript's current Fig. 1 as its overview, adds a project avatar, and reorganizes the README around the scientific problem, method, and reproducibility paths.
- **[2026-07-11]** The repository documents a reference-conditioned view of perturbation prediction: an unseen perturbation requires both a perturbation-specific shift and a compatible starting cell state.

---

## 为什么选择 TriShift？

单细胞扰动实验揭示了细胞如何响应基因编辑、药物、细胞因子和组合条件，但实验空间会随扰动、剂量、时间、细胞状态和个体背景迅速膨胀。仅凭“扰动是什么”并不能决定预测应从哪一种细胞状态开始。

TriShift 将这个问题拆成三个相互依赖的部分：

- **找对起点：** 用最优传输从未配对观测中构建状态兼容的参考候选，而不把它误解为真实的一对一细胞配对。
- **表示未知扰动：** 结合外部扰动先验，帮助模型在训练中没有直接观测的扰动上形成可迁移表示。
- **生成条件响应：** 从参考细胞出发，预测参考相对的表达位移，同时保留群体异质性而非只回归平均终点。

这使 TriShift 适用于严格的未见扰动、参考转移、组合泛化与分布恢复评估；不同指标衡量不同失败模式，因此仓库同时保留端点、响应基因与分布层面的诊断。

## 方法概览

<p align="center">
  <img src="assets/readme/trishift-fig1.png" alt="TriShift manuscript Figure 1: experimental data, representation learning, OT reference retrieval, shift prediction, and reference-conditioned generation" width="100%" />
</p>

*Current manuscript Fig. 1. TriShift learns a retrieval representation, constructs OT-guided state-compatible reference pools, predicts a perturbation-specific shift, and generates a reference-conditioned response.*

## 从这里开始

| 你的目标 | 推荐入口 |
| --- | --- |
| 快速验证核心训练/预测路径 | [Adamson mini demo](#built-in-adamson-mini-demo) |
| 用自己的 `AnnData` 运行 TriShift | [custom-dataset tutorial](#minimal-custom-dataset-tutorial) |
| 复现实验、基线与论文图 | [reproducibility guide](#for-reproducibility) |
| 查看论文源文件与编译产物 | [TriShift paper repository](https://github.com/elan6666/trishift-paper) |

---

## For Users

TriShift is a single-cell perturbation response prediction toolkit built around
the `Tripartite Reference-Conditioned Shift Model`. The repository contains the
native implementation, shared evaluation code, baseline wrappers, and the
notebooks used to generate manuscript figures. It uses a `src/` layout and is
installable as a Python package.

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

### Built-in Adamson mini demo

The repository includes a tiny Adamson-derived smoke-test dataset that is small enough to ship with GitHub:

- `examples/adamson_mini`

Run it after installing the package:

```bash
python examples/adamson_mini/run_demo.py
```

This demo trains and evaluates a 10% Adamson subset with Adamson-like settings, `1` split, and `20` epochs, then writes outputs to `artifacts/demo/adamson_mini`. It is meant to validate the code path, not to reproduce paper metrics.

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
python scripts/data/download_repro_inputs.py --items benchmark genept
```

This entrypoint delegates raw data download to `GEARS/PertData`, prepares the standard simulation splits, synchronizes `perturb_processed.h5ad` files to the paths expected by TriShift and the evaluation wrappers, and downloads the default GenePT embedding.
Run this command in an environment that has `GEARS/PertData` installed. The core `pip install -e .` environment is enough for TriShift package imports, but the public benchmark downloader needs the baseline-oriented environment described below.
The maintained public benchmark scope in this repository is `adamson`, `dixit`, `norman`, plus the scGen PBMC IFN-beta cell-type transfer case described below.

By default, the repository expects local data under `src/data`. You can still override locations through:

- `configs/paths.yaml`

`src/data` is intentionally ignored by git. It is a local cache for downloaded datasets, processed `.h5ad` files, and embedding files; do not rely on files under `src/data` as repository entrypoints. Use the maintained script above for reproducible data preparation.

### scGen PBMC IFN-beta case study

The scGen PBMC case uses the Kang IFN-beta PBMC file distributed through the
[scGen perturbation-prediction tutorial](https://scgen.readthedocs.io/en/stable/tutorials/scgen_perturbation_prediction.html)
as `train_kang.h5ad` (backup file id `1r87vhoLLq6PXAYdmyyd89zG90eJOFYLk`).
Download it to the expected local path with:

```bash
python scripts/data/download_repro_inputs.py --items scgen genept
```

If you want all protein-prior variants, include protein assets too:

```bash
python scripts/data/download_repro_inputs.py --items scgen genept protein
```

The unified downloader writes:

- `src/data/scgen/train_kang_scgen.h5ad`
- `src/data/scgen/perturb_processed.h5ad`

It also extracts the IFNB1/IFN-beta perturbation prior for the single `stimulated` condition and writes four switchable prior files under:

- `src/data/scgen/priors`

Prior extraction expects the local protein embedding files under `src/data/protein_embeddings` and the GenePT file under `src/data/Data_GeneEmbd`. Use `--skip-scgen-priors` if you only need to regenerate the `.h5ad`.

The TriShift entrypoint is:

```bash
python scripts/trishift/scgen_pbmc_celltype/run_scgen_pbmc_celltype.py
```

External baseline entrypoints for the same scGen PBMC split are:

```bash
python scripts/biolord/scgen_pbmc_celltype/run_biolord_scgen_pbmc_celltype.py
python scripts/scgpt/scgen_pbmc_celltype/run_scgpt_scgen_pbmc_celltype.py
```

This experiment holds out cell types, not perturbations: the only perturbation is `stimulated`, so the model trains on control and `stimulated` cells from seen cell types, withholds target-domain control and stimulated cells from training/model selection, and uses the target-domain control cells only as test-time reference origins.
To switch the prior, edit `defaults_overrides.emb_key` in:

- `scripts/trishift/scgen_pbmc_celltype/config.yaml`

Supported keys are:

- `emb_scgen_ifnb1_uniprot_prott5`
- `emb_scgen_ifnb1_zenodo_prott5`
- `emb_scgen_ifnb1_esm2_15b`
- `emb_scgen_ifnb1_genept`

BioLORD uses the same four external IFNB1 prior keys through `task_args.prior_key` in:

- `scripts/biolord/scgen_pbmc_celltype/config.yaml`

BioLORD also supports `biolord_self_attribute`, which uses a generated scalar attribute (`ctrl=0`, `stimulated=1`) instead of an external prior.

## For Reproducibility

For the complete paper workflow, including the order of TriShift, baseline, Systema, and notebook runs, see:

- `REPRODUCIBILITY_QUICK.md` for a short smoke-test and TriShift-only path
- `REPRODUCIBILITY.md`

### Quick reproduction paths

Use one of the following three scopes depending on what you need to verify.

Most local inputs can be downloaded or prepared through the unified entrypoint:

```bash
python scripts/data/download_repro_inputs.py --items benchmark genept
```

For the full local input setup, including optional protein embeddings, scGPT checkpoint files, scGen PBMC, and BioLORD-prepared h5ad files:

```bash
pip install gdown
python scripts/data/download_repro_inputs.py --items all --check
```

This command can take a long time and downloads several large files. Use `--items` to fetch only the groups you need.

1. Package smoke test

```bash
pip install -e .
python examples/adamson_mini/run_demo.py
```

This validates the core `TriShiftData -> TriShift train/evaluate -> saved outputs` path without requiring the public benchmark stack.

2. Public benchmark reproduction

```bash
python scripts/data/download_repro_inputs.py --items benchmark genept
python scripts/trishift/adamson/run_adamson.py
python scripts/trishift/dixit/run_dixit.py
python scripts/trishift/norman/run_norman.py
```

This reproduces the maintained TriShift benchmark scope in this repository and writes model outputs under `artifacts/results/<dataset>`, for example `artifacts/results/adamson`.

For the scGen PBMC case study:

```bash
python scripts/data/download_repro_inputs.py --items scgen protein genept
python scripts/trishift/scgen_pbmc_celltype/run_scgen_pbmc_celltype.py
python scripts/biolord/scgen_pbmc_celltype/run_biolord_scgen_pbmc_celltype.py
python scripts/scgpt/scgen_pbmc_celltype/run_scgpt_scgen_pbmc_celltype.py
```

3. Paper-figure regeneration

After the required baseline and Systema result folders exist, execute the figure notebooks listed in `REPRODUCIBILITY.md`. Primary figure artifacts are written under:

- `artifacts/paper_figures/main`
- `artifacts/paper_figures/supp`

The standalone manuscript source, compiled PDF, and supplementary document are maintained separately in:

- <https://github.com/elan6666/trishift-paper>

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

`environment_baselines.yml` covers the common stack used by `GEARS` and shared evaluation tools. `GEARS` still requires a Torch/PyG installation matched to your local CUDA runtime; follow the comments in that file for the final install step.

### Local asset classes

This repository keeps three local asset classes separate:

| Asset class | Managed by | Default location | Examples |
| --- | --- | --- | --- |
| Data and prepared inputs | `scripts/data/download_repro_inputs.py` and dataset-specific helpers under `scripts/data/` | `src/data/` | benchmark `.h5ad`, GenePT/protein priors, scGen PBMC inputs, BioLORD prepared `.h5ad` |
| External source trees | `scripts/setup/bootstrap_external_baselines.py` | `external/` | scGPT, GEARS, BioLORD, GenePert, CellOT |
| Model checkpoints | `scripts/data/download_repro_inputs.py` when a stable source is available, otherwise `configs/paths.yaml` | `artifacts/models/` | scGPT whole-human checkpoint |

For scGPT specifically, keep the source checkout and pretrained checkpoint separate:

- source tree: `external/scGPT-main`
- checkpoint files: `artifacts/models/scGPT_human/{args.json,best_model.pt,vocab.json}`

### External baseline source trees

Baseline repositories are not tracked directly because `external/` is a local, ignored workspace for third-party source trees, generated caches, and large intermediate files. To populate the external baselines and apply the tracked TriShift compatibility overlays, run:

```bash
python scripts/setup/bootstrap_external_baselines.py --only scgpt,gears,biolord,genepert,cellot
```

If you already downloaded the baseline repositories, copy from that folder instead:

```bash
python scripts/setup/bootstrap_external_baselines.py --source-root /path/to/downloads --force
```

The script places sources under `external/` and applies tracked overlays from `patches/external_overlays`. The current overlays include scGPT flash-attention compatibility files.
Scouter is treated as an optional local result/source cache in this repository. If you use Scouter outputs, place or sync them under `external/scouter/`; the bootstrap script does not clone Scouter because this workflow currently consumes only already generated Scouter result files.

This bootstrap step prepares source trees only. You still need the matching conda/pip environment for each baseline before running its training script.
For the scGen PBMC baselines, `scripts/biolord/...` imports the installed BioLORD package while reading the local scGen `.h5ad` and IFNB1 prior files prepared by `scripts/data/prepare_scgen_pbmc.py`; `scripts/scgpt/...` uses the local scGPT source tree and pretrained checkpoint.

### Reproduction input checker

After downloading data and external assets, check the expected local files with:

```bash
python scripts/setup/check_repro_inputs.py --scope benchmark --strict
python scripts/setup/check_repro_inputs.py --scope baselines
python scripts/setup/check_repro_inputs.py --scope scgen
```

The checker reports missing local datasets, embeddings, BioLORD inputs, external source trees, optional Scouter result cache, and scGPT checkpoint files.

### Data download and preprocessing

The preferred data entrypoint is:

```bash
python scripts/data/download_repro_inputs.py --items benchmark genept
```

Useful item groups:

| Item | What it does |
| --- | --- |
| `benchmark` | Downloads/prepares Adamson, Dixit, and Norman through GEARS/PertData and syncs `perturb_processed.h5ad` files. |
| `genept` | Downloads GenePT Zenodo archives and extracts `emb_b`, `emb_c`, and `emb_d`; use `--skip-legacy-genept` to skip the older `emb_c` archive. |
| `scgen` | Downloads Kang/scGen PBMC and runs `prepare_scgen_pbmc.py` unless `--no-prepare-scgen` is passed. |
| `protein` | Downloads UniProt ProtT5, Zenodo ProtT5, and Hugging Face ESM2 protein embeddings for IFNB1 prior variants. |
| `scgpt` | Downloads the scGPT whole-human pretrained checkpoint files into `artifacts/models/scGPT_human`. |
| `biolord` | Builds BioLORD-specific h5ad files from prepared benchmark data and GO graph inputs. |
| `all` | Runs all groups above. |

Examples:

```bash
python scripts/data/download_repro_inputs.py --items benchmark genept --check --check-scope benchmark
python scripts/data/download_repro_inputs.py --items scgen protein genept --skip-scgen-priors
python scripts/data/download_repro_inputs.py --items scgpt
```

Gene embeddings are external local artifacts and are not shipped with this repository. Download the required embedding files and place them under:

- `src/data/Data_GeneEmbd`

The default `configs/paths.yaml` expects the following files:

| Config key | Expected local file | Source |
| --- | --- | --- |
| `emb_a` | `src/data/Data_GeneEmbd/ensem_emb_gpt3.5all_new.pickle` | scELMo library, file `Gene-GPT 3.5`: <https://sites.google.com/yale.edu/scelmolib> |
| `emb_b` | `src/data/Data_GeneEmbd/GenePT_gene_embedding_ada_text.pickle` | GenePT Zenodo record: <https://zenodo.org/records/10833191> |
| `emb_c` | `src/data/Data_GeneEmbd/GPT_3_5_gene_embeddings.pickle` | GenePT Zenodo record: <https://zenodo.org/records/10030426> |
| `emb_d` | `src/data/Data_GeneEmbd/GenePT_gene_protein_embedding_model_3_text.pickle` | Optional GenePT protein/text embedding used only if selected in custom configs |

If your embedding files live elsewhere, update `configs/paths.yaml` or the dataset-specific config before training.

The benchmark preparation script builds the GEARS-native dataset folders under:

- `src/data/Data_GEARS/adamson`
- `src/data/Data_GEARS/dixit`
- `src/data/Data_GEARS/norman`

It also copies each generated `perturb_processed.h5ad` into the standard outer data directories:

- `src/data/adamson/perturb_processed.h5ad`
- `src/data/dixit/perturb_processed.h5ad`
- `src/data/norman/perturb_processed.h5ad`

This keeps the repository consistent across:

- GEARS, which reads from `src/data/Data_GEARS`
- TriShift and Systema-style evaluation, which read from `src/data/<dataset>`

For the scGen PBMC case, place the scGen-preprocessed Kang PBMC file at:

- `src/data/scgen/train_kang_scgen.h5ad`

The maintained downloader uses the scGen tutorial backup file for `train_kang.h5ad`:

```bash
python scripts/data/download_repro_inputs.py --items scgen genept
```

The script writes `src/data/scgen/perturb_processed.h5ad` and, unless `--skip-scgen-priors` is used, writes the IFNB1 prior pickle files under `src/data/scgen/priors`. The protein prior extraction expects:

- `src/data/protein_embeddings/uniprot_prott5_human_per_protein.h5`
- `src/data/protein_embeddings/zenodo_prott5_human_reduced_embeddings_file.h5`
- `src/data/protein_embeddings/hf_esm2_15b_human_mouse_embeddings.npy`
- `src/data/protein_embeddings/hf_esm2_15b_human_mouse_metadata.csv.gz`

The GenePT prior extraction expects:

- `src/data/Data_GeneEmbd/GenePT_gene_embedding_ada_text.pickle`

Protein embedding download sources used by `scripts/data/download_repro_inputs.py --items protein`:

| Local file | Source |
| --- | --- |
| `src/data/protein_embeddings/uniprot_prott5_human_per_protein.h5` | UniProt UP000005640_9606 per-protein ProtT5: <https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/embeddings/UP000005640_9606/per-protein.h5> |
| `src/data/protein_embeddings/zenodo_prott5_human_reduced_embeddings_file.h5` | Zenodo `10.5281/zenodo.5047020`, file `reduced_embeddings_file.h5` |
| `src/data/protein_embeddings/hf_esm2_15b_human_mouse_embeddings.npy` | Hugging Face `Darkadin/ESM2_embeddings_Human_Mouse`, file `ESM2_15B_Human_Mouse_Embeddings.npy` |
| `src/data/protein_embeddings/hf_esm2_15b_human_mouse_metadata.csv.gz` | Hugging Face `Darkadin/ESM2_embeddings_Human_Mouse`, file `ESM2_15B_Human_Mouse_Metadata.csv.gz` |

For scGPT, the benchmark wrappers expect the whole-human pretrained checkpoint under:

- `artifacts/models/scGPT_human/args.json`
- `artifacts/models/scGPT_human/best_model.pt`
- `artifacts/models/scGPT_human/vocab.json`

Download it with:

```bash
pip install gdown
python scripts/data/download_repro_inputs.py --items scgpt
```

The script uses the public Google Drive file ids distributed with the scGPT whole-human checkpoint. If you place the checkpoint elsewhere, update `scgpt_pretrained_root` in `configs/paths.yaml`.

### Training and evaluation entrypoints

Recommended dataset entrypoints are organized by model and dataset under `scripts/<model>/<dataset>`.
The maintained public targets are `adamson`, `dixit`, `norman`, and the scGen PBMC IFN-beta cell-type transfer case.

The maintained public interfaces for manuscript reproduction are:

- benchmark data preparation under `scripts/data`
- model entrypoints under `scripts/<model>/<dataset>`
- shared evaluation cores under `scripts/*/_core`
- figure notebooks under `notebooks`
- reproducibility instructions in `README.md` and `REPRODUCIBILITY.md`

TriShift:

- `scripts/trishift/adamson/run_adamson.py`
- `scripts/trishift/dixit/run_dixit.py`
- `scripts/trishift/norman/run_norman.py`
- `scripts/trishift/scgen_pbmc_celltype/run_scgen_pbmc_celltype.py`

GEARS:

- `scripts/gears/adamson/run_gears_adamson.py`
- `scripts/gears/dixit/run_gears_dixit.py`
- `scripts/gears/norman/run_gears_norman.py`

Additional baselines:

- `scripts/genepert/<dataset>/run_genepert_*.py`
- `scripts/scgpt/<dataset>/run_scgpt_*.py`
- `scripts/biolord/<dataset>/run_biolord_*.py`
- `scripts/systema/<dataset>/run_systema_*.py`

Shared training core:

- `scripts/trishift/_core/run_dataset_core.py`
- `scripts/trishift/train/run_dataset.py`

### Figure generation

The paper figures are generated from the notebooks under `notebooks/`.
These notebooks are the maintained figure-generation entrypoints; do not call a
standalone figure-rendering script to create manuscript images. Shared plotting
utilities live in `notebooks/_figure_helpers.py` and are imported by notebook
execution only.

- `Fig2_ReferenceTransfer.ipynb` -> Fig. 2
- `Fig3_Ablation.ipynb` -> Fig. 3
- `Fig4_NormanGeneralization.ipynb` -> Fig. 4
- `Fig5_DistributionRecovery.ipynb` -> Fig. 5
- `FigS1_ReferenceTransferFull.ipynb` -> Fig. S1
- `FigS2_AdditionalCases.ipynb` -> Fig. S2
- `FigS3_DixitRobustness.ipynb` -> Fig. S3
- `FigS4_Module1LatentState.ipynb` -> Fig. S4

Primary outputs are written under:

- `artifacts/results`
- `artifacts/paper_figures`

### Notes

- Legacy top-level `scripts/run_*` files, if present, should be treated as compatibility entrypoints rather than the primary maintained interface.
- Large local outputs, datasets, and external baseline clones are intentionally ignored by git.
