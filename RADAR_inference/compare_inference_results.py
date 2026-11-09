import argparse
import csv
import json
import math
from pathlib import Path
import re


def read_results(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames
        rows = {}
        for row in reader:
            name = row["file_name"]
            if name in rows:
                raise ValueError(f"Duplicate case in {path}: {name}")
            rows[name] = row
    return columns, rows


def main():
    parser = argparse.ArgumentParser(description="Compare inference scores and summarize worker timings")
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--worker-dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    columns, reference = read_results(args.reference)
    candidate_columns, candidate = read_results(args.candidate)
    if columns != candidate_columns:
        raise ValueError("CSV columns differ")
    if reference.keys() != candidate.keys():
        raise ValueError("Case coverage differs")
    compared = 0
    missing = 0
    missing_differences = 0
    score_differences = 0
    maximum_difference = 0.0
    for name, row in reference.items():
        for column in columns[1:]:
            expected, actual = row[column], candidate[name][column]
            if not expected or not actual:
                missing += not expected and not actual
                missing_differences += bool(expected) != bool(actual)
                continue
            expected_value, actual_value = float(expected), float(actual)
            if not math.isfinite(expected_value) or not math.isfinite(actual_value):
                raise ValueError(f"Nonfinite score: {name}, {column}")
            compared += 1
            score_differences += expected_value != actual_value
            maximum_difference = max(maximum_difference, abs(expected_value - actual_value))
    result = {
        "reference": str(args.reference), "candidate": str(args.candidate),
        "cases": len(candidate), "disease_columns": len(columns) - 1,
        "compared_scores": compared, "matching_missing_entries": missing,
        "missing_entry_differences": missing_differences,
        "score_differences": score_differences,
        "maximum_absolute_score_difference": maximum_difference,
    }
    if args.worker_dir:
        workers = []
        for path in sorted(args.worker_dir.glob("*.log")):
            content = path.read_text(encoding="utf-8", errors="replace")
            timings = [(float(wait), float(inference)) for wait, inference in re.findall(
                r"^  data wait seconds=([\d.]+), inference seconds=([\d.]+)", content, re.MULTILINE
            )]
            completion = re.findall(r"Saved (\d+) cases.*elapsed seconds=([\d.]+)", content)
            if not completion:
                raise ValueError(f"Worker has no completion record: {path}")
            count, elapsed = completion[-1]
            if len(timings) != int(count):
                raise ValueError(f"Worker timing coverage differs: {path}")
            workers.append({
                "worker": path.stem, "cases": int(count), "processing_seconds": float(elapsed),
                "data_wait_seconds": sum(wait for wait, _ in timings),
                "data_wait_after_first_seconds": sum(wait for wait, _ in timings[1:]),
                "inference_seconds": sum(inference for _, inference in timings),
            })
        result["workers"] = workers
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return int(bool(missing_differences or score_differences))


if __name__ == "__main__":
    raise SystemExit(main())
