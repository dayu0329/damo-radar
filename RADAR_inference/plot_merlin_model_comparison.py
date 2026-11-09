import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


ROOT = Path(__file__).resolve().parents[1]
MODELS = (
    ("merlin_metrics_radar_pretrain", "RADAR pretrain", "red", ROOT / "results/RADAR_infer_results_MerlinTestset.csv"),
    ("merlin_metrics_radar_plus", "RADAR+", "green", ROOT / "results/merlin_testset_radar_plus/RADAR_infer_results_radar_plus_merlin_testset.csv"),
    ("merlin_metrics_epoch29", "Epoch29", "blue", ROOT / "results/merlin_testset_epoch29/RADAR_infer_results_epoch29_merlin_testset.csv"),
)


def metric_table(folder):
    frame = pd.read_csv(ROOT / "results" / folder / "metrics_per_disease.csv", encoding="utf-8-sig")
    frame.columns = [column.split("（", 1)[0] for column in frame.columns]
    return frame.loc[frame["disease"].notna()].set_index("disease")


def patient_id(filename):
    name = str(filename).replace("\\", "/").rsplit("/", 1)[-1]
    for suffix in (".nii.gz", ".nii"):
        if name.endswith(suffix):
            return name[:-len(suffix)]
    return name


def main():
    parser = argparse.ArgumentParser(description="Compare MERLIN AUROC and ROC curves for three models")
    parser.add_argument("--labels", type=Path, default=Path("/data/dataset/json/merlin_labels.json"))
    parser.add_argument("--font", type=Path, default=Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/merlin_model_comparison")
    args = parser.parse_args()
    labels = json.loads(args.labels.read_text(encoding="utf-8-sig"))
    tables = {name: metric_table(folder) for folder, name, _, _ in MODELS}
    diseases = list(tables[MODELS[0][1]].index)
    diseases = [disease for disease in diseases if all(disease in table.index for table in tables.values())]
    if len(diseases) != 20:
        raise ValueError(f"Expected 20 shared diseases; found {len(diseases)}")
    chinese = [str(tables[MODELS[0][1]].loc[disease, "disease_cn"]).split("_", 1)[-1] for disease in diseases]
    font = font_manager.FontProperties(fname=str(args.font))
    plt.rcParams["axes.unicode_minus"] = False
    curves = {}
    for _, name, _, prediction_path in MODELS:
        predictions = pd.read_csv(prediction_path, encoding="utf-8-sig")
        if not predictions["file_name"].is_unique:
            raise ValueError(f"Duplicate filenames in {prediction_path}")
        curves[name] = {}
        for disease in diseases:
            column = tables[name].loc[disease, "column"]
            targets = []
            scores = []
            for filename, prediction in zip(predictions["file_name"], predictions[column]):
                label = labels[disease].get(patient_id(filename))
                try:
                    label = float(label)
                except (TypeError, ValueError):
                    continue
                if label not in (0, 1):
                    continue
                score = 0.0 if pd.isna(prediction) else float(prediction)
                if not np.isfinite(score):
                    score = 0.0
                targets.append(int(label))
                scores.append(score)
            if len(set(targets)) != 2:
                raise ValueError(f"ROC requires both classes for {name}: {disease}")
            fpr, tpr, _ = roc_curve(targets, scores)
            auc = roc_auc_score(targets, scores)
            reference = float(tables[name].loc[disease, "roc_auc"])
            if not np.isclose(auc, reference, atol=1e-12):
                raise ValueError(f"ROC AUC differs from saved metric for {name}: {disease}")
            curves[name][disease] = (fpr, tpr, auc)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(21, 9))
    x = np.arange(21)
    for index, (_, name, color, _) in enumerate(MODELS):
        values = [curves[name][disease][2] for disease in diseases]
        values.append(float(np.mean(values)))
        bars = axis.bar(x + (index - 1) * 0.25, values, 0.25, label=name, color=color, alpha=0.82)
        for bar, value in zip(bars, values):
            axis.text(bar.get_x() + bar.get_width() / 2, value + 0.008, f"{value:.3f}", ha="center", fontsize=7, rotation=90)
    axis.set_xticks(x, [*chinese, "平均"], fontproperties=font, rotation=45, ha="right")
    axis.set_ylim(0, 1.12)
    axis.set_ylabel("AUROC")
    axis.set_title("MERLIN 测试集：20 种疾病及平均 AUROC", fontproperties=font)
    axis.legend()
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(args.output_dir / "auc_20_diseases_and_mean.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(4, 5, figsize=(24, 20), constrained_layout=True)
    for axis, disease, title in zip(axes.flat, diseases, chinese):
        for _, name, color, _ in MODELS:
            fpr, tpr, auc = curves[name][disease]
            axis.plot(fpr, tpr, color=color, lw=2, label=f"{name}  {auc:.3f}")
        axis.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1)
        axis.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="1 - Specificity", ylabel="Sensitivity")
        axis.set_title(title, fontproperties=font)
        axis.grid(alpha=0.2)
        axis.legend(fontsize=7, loc="lower right")
    fig.suptitle("MERLIN 测试集：三模型 ROC 对比", fontproperties=font, fontsize=18)
    fig.savefig(args.output_dir / "roc_20_diseases_three_models.jpg", dpi=180)
    plt.close(fig)
    print(f"Saved comparison figures to {args.output_dir}")


if __name__ == "__main__":
    main()
