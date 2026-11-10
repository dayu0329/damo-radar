"""Compatibility entry for older scripts; use inference_merlin_testset_all.py."""

from pathlib import Path

from inference_merlin_testset_all import main as run_inference

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = ROOT / "training_results/gradcache48/20261102_201102_310236/checkpoint_9.pth"


def main(defaults=None):
    overrides = dict(defaults or {})
    checkpoint_name = Path(overrides.get("checkpoint", DEFAULT_CHECKPOINT)).name
    overrides.setdefault(
        "model_profile",
        "radar_plus" if checkpoint_name == "checkpoint_radar_plus.pth"
        else "epoch29" if checkpoint_name == "checkpoint_29.pth" else "epoch10",
    )
    overrides.setdefault(
        "dataset_profile",
        "merlin_testset" if overrides.get("report_json") else "yg_images",
    )
    return run_inference(overrides)


if __name__ == "__main__":
    main()
