import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates-per-epoch", type=int, default=318)
    args = parser.parse_args()
    lines = args.metrics.read_text().splitlines()
    rows = []
    for index, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if index != len(lines) - 1:
                raise
    epochs = defaultdict(list)
    for row in rows:
        epochs[row["epoch"]].append(row)
    summaries = []
    for epoch, records in sorted(epochs.items()):
        result = {"round": epoch + 1, "updates": len(records), "complete": len(records) == args.updates_per_epoch}
        for branch in ("global", "organ"):
            selected = [record for record in records if record["branch"] == branch]
            if not selected:
                continue
            values = {}
            for key in ("loss_itc", "loss_itc_per_organ", "loss_seg", "retrieval_top1", "similarity_gap", "image_feature_std", "text_feature_std", "temperature", "gradient_norm", "step_seconds", "load_seconds"):
                numbers = [record[key] for record in selected if key in record and record[key] is not None]
                if numbers:
                    values[key] = mean(numbers)
            if branch == "organ":
                for key in ("retrieval_top1", "similarity_gap"):
                    numbers = [organ[key] for record in selected for organ in record["organs"].values()]
                    values[key] = mean(numbers) if numbers else None
            result[branch] = values
        result["skipped"] = sum(record["update_skipped"] for record in records)
        result["peak_mib"] = max(record["peak_allocated_mib"] for record in records)
        result["mean_update_seconds"] = mean(record["step_seconds"] for record in records)
        summaries.append(result)
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"latest": rows[-1]["timestamp"], "updates": len(rows), "source": str(args.metrics), "epochs": summaries}
    (args.output / "epoch_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    figure, axes = plt.subplots(2, 3, figsize=(15, 8.5), constrained_layout=True)
    panels = [
        (axes[0, 0], "Whole-image contrastive loss", "global", "loss_itc", 1),
        (axes[0, 1], "Whole-image retrieval top-1 (%)", "global", "retrieval_top1", 100),
        (axes[0, 2], "Whole-image positive-negative cosine gap", "global", "similarity_gap", 1),
        (axes[1, 0], "Contrastive loss per eligible organ", "organ", "loss_itc_per_organ", 1),
        (axes[1, 1], "Organ retrieval diagnostic (%)", "organ", "retrieval_top1", 100),
        (axes[1, 2], "Organ positive-negative cosine gap", "organ", "similarity_gap", 1),
    ]
    for axis, title, branch, key, scale in panels:
        selected = [entry for entry in summaries if entry.get(branch, {}).get(key) is not None]
        axis.plot([entry["round"] for entry in selected], [entry[branch][key] * scale for entry in selected], "o-", color="#2563eb", markersize=4)
        for entry in selected:
            if not entry["complete"]:
                axis.scatter(entry["round"], entry[branch][key] * scale, marker="D", s=65, facecolor="#f59e0b", edgecolor="#92400e", zorder=5)
        axis.set_title(title)
        axis.set_xlabel("Training round (1-based)")
        axis.grid(alpha=0.2)
        axis.set_xticks(range(1, len(summaries) + 1, 2))
    axes[0, 0].axhline(3.871201, color="#888888", linestyle=":", label="ln(48) random baseline")
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].axhline(100 / 48, color="#888888", linestyle=":")
    figure.suptitle(f"RADAR training metrics | {rows[-1]['timestamp'][:19]} CST\nRound means; orange diamond = incomplete round. Training diagnostics, not validation performance.", fontsize=13)
    figure.savefig(args.output / "learning_curves.png", dpi=160)
    plt.close(figure)
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
    for branch, color in (("global", "#2563eb"), ("organ", "#d97706")):
        selected = [entry for entry in summaries if branch in entry]
        for axis, key, title in zip(axes, ("loss_seg", "gradient_norm", "step_seconds"), ("Segmentation loss", "Gradient norm (no clipping)", "Seconds per optimizer update")):
            axis.plot([entry["round"] for entry in selected], [entry[branch][key] for entry in selected], "o-", color=color, label=branch, markersize=4)
            axis.set_title(title)
            axis.set_xlabel("Training round (1-based)")
            axis.grid(alpha=0.2)
            axis.legend()
    figure.suptitle("Optimization and throughput | incomplete final round included", fontsize=13)
    figure.savefig(args.output / "optimization_curves.png", dpi=160)
    plt.close(figure)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
