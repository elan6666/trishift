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
    {"figure": "Fig2", "panel": "h", "description": "reference-transfer gene-level case if space allows", "source": "payload_pkl", "notebook": "notebooks/Fig2_ReferenceTransfer.ipynb"},
    {"figure": "Fig3", "panel": "a", "description": "reference construction ablation: random/kNN/OT", "source": "artifacts/results/ablation/*/*/metrics_unseen_ctrl.csv", "notebook": "notebooks/Fig3_Ablation.ipynb"},
    {"figure": "Fig3", "panel": "b", "description": "conditioning input ablation: no reference/no prior/full", "source": "artifacts/results/ablation/*/*/metrics_unseen_ctrl.csv", "notebook": "notebooks/Fig3_Ablation.ipynb"},
    {"figure": "Fig3", "panel": "c", "description": "shift/fusion ablation: MLP/transformer/full", "source": "artifacts/results/ablation/*/*/metrics_unseen_ctrl.csv", "notebook": "notebooks/Fig3_Ablation.ipynb"},
    {"figure": "Fig3", "panel": "d", "description": "expression objective ablation: L_mse/L_gears/L_gears+L_neg", "source": "artifacts/results/ablation/*/*/metrics_unseen_ctrl.csv", "notebook": "notebooks/Fig3_Ablation.ipynb"},
    {"figure": "Fig4", "panel": "a", "description": "Norman subgroup heatmap", "source": "metrics_long", "notebook": "notebooks/Fig4_NormanGeneralization.ipynb"},
    {"figure": "Fig4", "panel": "b", "description": "Norman seen0 core metrics", "source": "metrics_long", "notebook": "notebooks/Fig4_NormanGeneralization.ipynb"},
    {"figure": "Fig4", "panel": "c", "description": "Norman seen0 DEG prediction / direction agreement", "source": "artifacts/analysis/deg_prediction/*/deg_prediction_long.csv", "notebook": "notebooks/Fig4_NormanGeneralization.ipynb"},
    {"figure": "Fig4", "panel": "d", "description": "Norman combination gene-level case", "source": "payload_pkl", "notebook": "notebooks/Fig4_NormanGeneralization.ipynb"},
    {"figure": "Fig5", "panel": "a", "description": "Wasserstein distance", "source": "distribution_metrics_long", "notebook": "notebooks/Fig5_DistributionRecovery.ipynb"},
    {"figure": "Fig5", "panel": "b", "description": "mean rho2", "source": "distribution_metrics_long", "notebook": "notebooks/Fig5_DistributionRecovery.ipynb"},
    {"figure": "Fig5", "panel": "c", "description": "variance rho2", "source": "distribution_metrics_long", "notebook": "notebooks/Fig5_DistributionRecovery.ipynb"},
    {"figure": "Fig5", "panel": "d", "description": "cell-level distribution case with CellOT", "source": "payload_pkl", "notebook": "notebooks/Fig5_DistributionRecovery.ipynb"},
    {"figure": "FigS1", "panel": "a-d", "description": "full reference-transfer distributions including DEG", "source": "Fig2 source long tables", "notebook": "notebooks/FigS1_ReferenceTransferFull.ipynb"},
    {"figure": "FigS2", "panel": "a-e", "description": "Dixit scarce-condition stability", "source": "metrics_long and DEG long tables", "notebook": "notebooks/FigS2_DixitStability.ipynb"},
    {"figure": "FigS3", "panel": "a-d", "description": "conventional unseen benchmark backup and baseline fairness", "source": "default benchmark long tables", "notebook": "notebooks/FigS3_UnseenBenchmarkBackup.ipynb"},
    {"figure": "FigS4", "panel": "a-g", "description": "ablation extensions and full Systema diagnostics", "source": "ablation metrics and systema long tables", "notebook": "notebooks/FigS4_AblationAndSystemaDiagnostics.ipynb"},
    {"figure": "FigS5", "panel": "a-f", "description": "combination and distribution extensions", "source": "Norman and distribution long tables", "notebook": "notebooks/FigS5_CombinationDistributionExtension.ipynb"},
    {"figure": "FigS6", "panel": "a-e", "description": "Module 1 PBMC latent-state analysis", "source": "stage1_latent_clustering outputs", "notebook": "notebooks/FigS6_Module1LatentState.ipynb"},
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
