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
    {"figure": "Fig2", "panel": "d", "description": "residualized Systema Pearson", "source": "artifacts/analysis/systema_mechanism/*/systema_mechanism_long.csv", "notebook": "notebooks/Fig2_ReferenceTransfer.ipynb"},
    {"figure": "Fig2", "panel": "e", "description": "centroid accuracy", "source": "artifacts/analysis/systema_mechanism/*/systema_mechanism_long.csv", "notebook": "notebooks/Fig2_ReferenceTransfer.ipynb"},
    {"figure": "Fig2", "panel": "f", "description": "train-distance / generic-shift dependence", "source": "artifacts/analysis/systema_mechanism/*", "notebook": "notebooks/Fig2_ReferenceTransfer.ipynb"},
    {"figure": "Fig2", "panel": "g", "description": "post-perturbation DEG AUROC/AUPRC", "source": "artifacts/analysis/deg_prediction/*/deg_prediction_long.csv", "notebook": "notebooks/Fig2_ReferenceTransfer.ipynb"},
    {"figure": "Fig2", "panel": "h", "description": "fixed ELMSAN1+ctrl gene-level response case", "source": "held-out-control payload PKLs or recovered case preview", "notebook": "notebooks/Fig2_ReferenceTransfer.ipynb"},
    {"figure": "Fig3", "panel": "a", "description": "reference construction ablation: Pearson for random/kNN/OT", "source": "artifacts/results/ablation/*/*/metrics_unseen_ctrl.csv plus PBMC metrics.csv", "notebook": "notebooks/Fig3_Ablation.ipynb"},
    {"figure": "Fig3", "panel": "b", "description": "reference construction ablation: nMSE for random/kNN/OT", "source": "artifacts/results/ablation/*/*/metrics_unseen_ctrl.csv plus PBMC metrics.csv", "notebook": "notebooks/Fig3_Ablation.ipynb"},
    {"figure": "Fig3", "panel": "c", "description": "reference construction ablation: Systema Pearson for random/kNN/OT", "source": "artifacts/results/ablation/*/*/metrics_unseen_ctrl.csv plus PBMC metrics.csv", "notebook": "notebooks/Fig3_Ablation.ipynb"},
    {"figure": "Fig3", "panel": "d", "description": "conditioning-input ablation: Pearson for no-reference/no-prior/full", "source": "artifacts/results/ablation/*/*/metrics_unseen_ctrl.csv plus PBMC metrics.csv", "notebook": "notebooks/Fig3_Ablation.ipynb"},
    {"figure": "Fig3", "panel": "e", "description": "conditioning-input ablation: nMSE for no-reference/no-prior/full", "source": "artifacts/results/ablation/*/*/metrics_unseen_ctrl.csv plus PBMC metrics.csv", "notebook": "notebooks/Fig3_Ablation.ipynb"},
    {"figure": "Fig3", "panel": "f", "description": "conditioning-input ablation: Systema Pearson for no-reference/no-prior/full", "source": "artifacts/results/ablation/*/*/metrics_unseen_ctrl.csv plus PBMC metrics.csv", "notebook": "notebooks/Fig3_Ablation.ipynb"},
    {"figure": "Fig4", "panel": "a", "description": "Norman held-out-control subgroup heatmap", "source": "held-out-control metrics_long", "notebook": "notebooks/Fig4_NormanGeneralization.ipynb"},
    {"figure": "Fig4", "panel": "b", "description": "Norman subgroup Pearson grouped bars", "source": "held-out-control metrics_long", "notebook": "notebooks/Fig4_NormanGeneralization.ipynb"},
    {"figure": "Fig4", "panel": "c", "description": "Norman subgroup nMSE grouped bars", "source": "held-out-control metrics_long", "notebook": "notebooks/Fig4_NormanGeneralization.ipynb"},
    {"figure": "Fig4", "panel": "d", "description": "Norman subgroup Systema Pearson grouped bars", "source": "held-out-control metrics_long", "notebook": "notebooks/Fig4_NormanGeneralization.ipynb"},
    {"figure": "Fig4", "panel": "e", "description": "Norman DEG prediction grouped bars", "source": "artifacts/analysis/deg_prediction/*/deg_prediction_long.csv", "notebook": "notebooks/Fig4_NormanGeneralization.ipynb"},
    {"figure": "Fig4", "panel": "f", "description": "Norman response-direction agreement grouped bars", "source": "artifacts/analysis/deg_prediction/*/deg_prediction_long.csv", "notebook": "notebooks/Fig4_NormanGeneralization.ipynb"},
    {"figure": "Fig4", "panel": "g", "description": "fixed CNN1+MAPK1 gene-level response case", "source": "held-out-control payload PKLs or recovered case preview", "notebook": "notebooks/Fig4_NormanGeneralization.ipynb"},
    {"figure": "Fig5", "panel": "a", "description": "Wasserstein distance", "source": "distribution_metrics_long", "notebook": "notebooks/Fig5_DistributionRecovery.ipynb"},
    {"figure": "Fig5", "panel": "b", "description": "mean rho2", "source": "distribution_metrics_long", "notebook": "notebooks/Fig5_DistributionRecovery.ipynb"},
    {"figure": "Fig5", "panel": "c", "description": "variance rho2", "source": "distribution_metrics_long", "notebook": "notebooks/Fig5_DistributionRecovery.ipynb"},
    {"figure": "Fig5", "panel": "d", "description": "RPS29 expression distribution under PTDSS1+ctrl", "source": "held-out-control payload PKLs or recovered additional-case panel", "notebook": "notebooks/Fig5_DistributionRecovery.ipynb"},
    {"figure": "FigS1", "panel": "a-g", "description": "boxplot counterpart to Fig. 2a-g", "source": "Fig2 source long tables", "notebook": "notebooks/FigS1_ReferenceTransferFull.ipynb"},
    {"figure": "FigS2", "panel": "a-e", "description": "Dixit held-out-control scarce-condition stability", "source": "held-out-control metrics_long and DEG long tables", "notebook": "notebooks/FigS2_DixitStability.ipynb"},
    {"figure": "FigS3", "panel": "a-e", "description": "additional fixed-display response cases restored from the legacy supplement plan", "source": "held-out-control payload PKLs or recovered FigS2_AdditionalCases panels", "notebook": "notebooks/FigS3_AdditionalCases.ipynb"},
    {"figure": "FigS4", "panel": "a-f", "description": "Dixit distance-stratified robustness diagnostics", "source": "held-out-control metrics, Systema mechanism metadata and DEG prediction tables", "notebook": "notebooks/FigS4_DixitRobustness.ipynb"},
    {"figure": "FigS5", "panel": "a-e", "description": "Module 1 PBMC latent-state analysis moved from the previous Fig. S6 slot", "source": "stage1_latent_clustering outputs", "notebook": "notebooks/FigS5_Module1LatentState.ipynb"},
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
