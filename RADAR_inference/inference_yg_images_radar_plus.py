import csv
import json
import math
from pathlib import Path

from inference_yg_images_epoch10 import ROOT, main as run_inference


def compare_results(epoch10_path, official_path):
    with epoch10_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames
        reference_rows = list(reader)
    reference = {row["file_name"]: row for row in reference_rows}
    if len(reference) != len(reference_rows):
        raise ValueError("Duplicate filenames in epoch-10 results")
    with official_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != columns:
            raise ValueError("Disease columns differ between the two result files")
        official_rows = list(reader)
    if len({row["file_name"] for row in official_rows}) != len(official_rows):
        raise ValueError("Duplicate filenames in official results")
    reference_meta = json.loads(epoch10_path.with_suffix(".json").read_text(encoding="utf-8"))
    official_meta = json.loads(official_path.with_suffix(".json").read_text(encoding="utf-8"))
    for setting in ("protocol", "disease_items", "aggregation", "clip_hu", "reference_spacing", "score_type"):
        if reference_meta[setting] != official_meta[setting]:
            raise ValueError(f"Inference settings differ: {setting}")
    feature_metadata = []
    for metadata in (reference_meta, official_meta):
        feature_path = Path(metadata["text_embeddings"])
        feature_metadata.append(json.loads(feature_path.with_suffix(feature_path.suffix + ".json").read_text(encoding="utf-8")))
    if feature_metadata[0]["prompts"] != feature_metadata[1]["prompts"]:
        raise ValueError("Prompt sentences differ between the two models")
    rows = []
    counts = {status: 0 for status in ("both_scored", "epoch10_only", "radar_plus_only", "neither_scored")}
    for official in official_rows:
        filename = official["file_name"]
        if filename not in reference:
            raise ValueError(f"No epoch-10 result for {filename}")
        for disease in columns[1:]:
            trained_score = float(reference[filename][disease]) if reference[filename][disease] else None
            official_score = float(official[disease]) if official[disease] else None
            for score in (trained_score, official_score):
                if score is not None and (not math.isfinite(score) or not 0 <= score <= 1):
                    raise ValueError(f"Invalid score for {filename}: {disease}")
            if trained_score is not None and official_score is not None:
                status = "both_scored"
                difference = trained_score - official_score
            else:
                status = "epoch10_only" if trained_score is not None else "radar_plus_only" if official_score is not None else "neither_scored"
                difference = None
            counts[status] += 1
            rows.append([filename, disease, trained_score, official_score, difference, status])
    output = official_path.with_name(f"{official_path.stem}_vs_epoch10.csv")
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["file_name", "disease", "epoch10_score", "radar_plus_score", "epoch10_minus_radar_plus", "status"])
        writer.writerows(rows)
    output.with_suffix(".json").write_text(json.dumps({
        "epoch10_results": str(epoch10_path.resolve()), "radar_plus_results": str(official_path.resolve()),
        "cases": len(official_rows), "pair_status_counts": counts,
        "ground_truth_available": False, "evaluation_metrics": None,
        "difference_definition": "epoch10_score - radar_plus_score; not a performance metric",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(rows)} paired items to {output}; statuses={counts}", flush=True)
    return output


def main():
    output = run_inference({
        "checkpoint": ROOT / "ckpt/checkpoint_radar_plus.pth",
        "text_embeddings": ROOT / "ckpt/infer_text_embedding_radar_plus_merlin_anatomy.pt",
        "bert_dir": ROOT / "ckpt/bert-base-uncased",
        "save_tag": "radar_plus_merlin",
    })
    if output is not None:
        reference = ROOT / "results/yg_images/RADAR_infer_results_epoch10_merlin.csv"
        if reference.exists():
            compare_results(reference, output)
        else:
            print(f"Comparison skipped: epoch-10 results not found at {reference}", flush=True)


if __name__ == "__main__":
    main()
