from pathlib import Path

from inference_yg_images_epoch10 import ROOT, main as run_inference


def main():
    return run_inference({
        "checkpoint": ROOT / "ckpt/checkpoint_radar_plus.pth",
        "text_embeddings": ROOT / "ckpt/infer_text_embedding_radar_plus_merlin_anatomy.pt",
        "bert_dir": ROOT / "ckpt/bert-base-uncased",
        "img_dir": Path("/data/dataset/other_images"),
        "report_json": Path("/data/dataset/json/merlin_report.json"),
        "save_dir": ROOT / "results/merlin_testset",
        "save_tag": "radar_plus_merlin_testset",
    })


if __name__ == "__main__":
    main()
