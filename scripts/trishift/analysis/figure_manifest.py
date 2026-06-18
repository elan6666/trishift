from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from scripts.common.time_utils import ts_local


FIGURE_ROWS: list[dict[str, Any]] = [
    {"figure": "Fig2", "panel": "a", "description": "held-out/reference-transfer Pearson", "source": "metrics_long", "notebook": "notebooks/Fig2_ReferenceTransfer.ipynb"},
    {"figure": "Fig2", "panel": "b", "description": "held-out/reference-transfer MSE or nMSE", "source": "metrics_long", "notebook": "notebooks/Fig2_ReferenceTransfer.ipynb"},
    {"figure": "Fig2", "panel": "c", "description": "held-out/reference-transfer Systema Pearson", "source": "metrics_long", "notebook": "notebooks/Fig2_ReferenceTransfer.ipynb"},
    {"figure": "Fig2", "panel": "d", "description": "residualized Systema Pearson", "source": "artifacts/analysis/systema_mechanism/*/systema_mechanism_long.csv plus PBMC target-domain metric fallback", "notebook": "notebooks/Fig2_ReferenceTransfer.ipynb"},
    {"figure": "Fig2", "panel": "e", "description": "centroid accuracy", "source": "artifacts/analysis/systema_mechanism/*/systema_mechanism_long.csv plus PBMC single-target centroid fallback", "notebook": "notebooks/Fig2_ReferenceTransfer.ipynb"},
    {"figure": "Fig2", "panel": "f", "description": "train-distance / generic-shift dependence", "source": "artifacts/analysis/systema_mechanism/* plus PBMC target-domain metric fallback", "notebook": "notebooks/Fig2_ReferenceTransfer.ipynb"},
    {"figure": "Fig2", "panel": "g", "description": "fixed ELMSAN1+ctrl gene-level response case", "source": "held-out-control payload PKLs or recovered case preview", "notebook": "notebooks/Fig2_ReferenceTransfer.ipynb"},
    {"figure": "Fig3", "panel": "a", "description": "reference construction ablation: nMSE for kNN/OT, grouped by dataset", "source": "mean_pearson*.txt summaries under artifacts/results/ablation plus PBMC mean_pearson.txt", "notebook": "notebooks/Fig3_Ablation.ipynb"},
    {"figure": "Fig3", "panel": "b", "description": "reference construction ablation: Systema Pearson for kNN/OT, grouped by dataset", "source": "mean_pearson*.txt summaries under artifacts/results/ablation plus PBMC mean_pearson.txt", "notebook": "notebooks/Fig3_Ablation.ipynb"},
    {"figure": "Fig3", "panel": "c", "description": "conditioning-input ablation: nMSE for no-reference/no-prior/full, grouped by dataset", "source": "mean_pearson*.txt summaries under artifacts/results/ablation plus PBMC mean_pearson.txt", "notebook": "notebooks/Fig3_Ablation.ipynb"},
    {"figure": "Fig3", "panel": "d", "description": "conditioning-input ablation: Systema Pearson for no-reference/no-prior/full, grouped by dataset", "source": "mean_pearson*.txt summaries under artifacts/results/ablation plus PBMC mean_pearson.txt", "notebook": "notebooks/Fig3_Ablation.ipynb"},
    {"figure": "Fig4", "panel": "a", "description": "Norman held-out-control subgroup heatmap", "source": "held-out-control metrics_long", "notebook": "notebooks/Fig4_NormanGeneralization.ipynb"},
    {"figure": "Fig4", "panel": "b", "description": "Norman subgroup Pearson grouped bars", "source": "held-out-control metrics_long", "notebook": "notebooks/Fig4_NormanGeneralization.ipynb"},
    {"figure": "Fig4", "panel": "c", "description": "Norman subgroup nMSE grouped bars", "source": "held-out-control metrics_long", "notebook": "notebooks/Fig4_NormanGeneralization.ipynb"},
    {"figure": "Fig4", "panel": "d", "description": "Norman subgroup Systema Pearson grouped bars", "source": "held-out-control metrics_long", "notebook": "notebooks/Fig4_NormanGeneralization.ipynb"},
    {"figure": "Fig4", "panel": "e", "description": "Norman Overlap@20 grouped bars for response-gene identity recovery", "source": "artifacts/analysis/deg_prediction/*/deg_prediction_long.csv", "notebook": "notebooks/Fig4_NormanGeneralization.ipynb"},
    {"figure": "Fig4", "panel": "f", "description": "fixed CNN1+MAPK1 gene-level response case", "source": "held-out-control payload PKLs or recovered case preview", "notebook": "notebooks/Fig4_NormanGeneralization.ipynb"},
    {"figure": "Fig5", "panel": "a", "description": "Wasserstein distance", "source": "distribution_metrics_long", "notebook": "notebooks/Fig5_DistributionRecovery.ipynb"},
    {"figure": "Fig5", "panel": "b", "description": "mean rho2", "source": "distribution_metrics_long", "notebook": "notebooks/Fig5_DistributionRecovery.ipynb"},
    {"figure": "Fig5", "panel": "c", "description": "variance rho2", "source": "distribution_metrics_long", "notebook": "notebooks/Fig5_DistributionRecovery.ipynb"},
    {"figure": "Fig5", "panel": "d", "description": "ISG15 expression distribution under PBMC IFN-beta stimulation across TriShift, scGPT and CellOT", "source": "artifacts/results/scgen_pbmc_celltype/trishift_scgen_pbmc_celltype_3.pkl; artifacts/results/scgpt/scgen_pbmc_celltype/scgpt_scgen_pbmc_celltype_3.pkl; artifacts/results/cellot/scgen_pbmc_celltype/cellot_scgen_pbmc_celltype_3_unseen_ctrl.pkl", "notebook": "notebooks/Fig5_DistributionRecovery.ipynb"},
    {"figure": "Fig5", "panel": "e", "description": "scDFM-style model-specific PBMC stimulated UMAP density panels for target-domain control, observed stimulated cells and predictions", "source": "artifacts/results/scgen_pbmc_celltype/trishift_scgen_pbmc_celltype_3.pkl; artifacts/results/scgpt/scgen_pbmc_celltype/scgpt_scgen_pbmc_celltype_3.pkl; artifacts/results/cellot/scgen_pbmc_celltype/cellot_scgen_pbmc_celltype_3_unseen_ctrl.pkl", "notebook": "notebooks/Fig5_DistributionRecovery.ipynb"},
    {"figure": "FigS1", "panel": "a-f", "description": "boxplot counterpart to Fig. 2a-f", "source": "Fig2 source long tables", "notebook": "notebooks/FigS1_ReferenceTransferFull.ipynb"},
    {"figure": "FigS2", "panel": "a-d", "description": "two additional fixed-display response cases plus one matched gene-expression distribution panel per case", "source": "held-out-control payload PKLs and recovered FigS2_AdditionalCases panels; Adamson/Norman/Dixit case panels exclude CellOT because CellOT is scoped to PBMC distributional comparisons in the manuscript figures", "notebook": "notebooks/FigS2_AdditionalCases.ipynb"},
    {"figure": "FigS3", "panel": "a-f", "description": "Dixit distance-stratified grouped-bar robustness diagnostics including Overlap@20", "source": "held-out-control metrics, Systema mechanism metadata and Overlap@20 response-gene identity tables", "notebook": "notebooks/FigS3_DixitRobustness.ipynb"},
    {"figure": "FigS4", "panel": "a-d", "description": "PBMC Module 1 latent-state diagnostics without Module 3 prediction UMAPs or external model comparisons", "source": "stage1_latent_clustering outputs from the PBMC Module 1 diagnostic run", "notebook": "notebooks/FigS4_Module1LatentState.ipynb"},
]


def write_manifest(out_dir: str | Path) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(FIGURE_ROWS)
    csv_path = out / "figure_manifest.csv"
    json_path = out / "figure_manifest.json"
    provenance_path = out / "provenance.json"
    df.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(FIGURE_ROWS, indent=2, ensure_ascii=False), encoding="utf-8")
    provenance_path.write_text(
        json.dumps(
            {
                "generated_at": ts_local(),
                "style": "compact, narrow, low-whitespace, soft palette, fine lines, small markers",
                "pathway_figure": "excluded",
                "cellot_in_fig3": False,
                "supplement_public_figures": "S1-S4",
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return {"csv": csv_path, "json": json_path, "provenance": provenance_path}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Write the refreshed TriShift paper figure manifest.")
    ap.add_argument("--out-dir", default="artifacts/paper_figures")
    args = ap.parse_args(argv)
    paths = write_manifest(args.out_dir)
    for key, path in paths.items():
        print(f"{key}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
