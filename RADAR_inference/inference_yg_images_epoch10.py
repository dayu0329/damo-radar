import argparse
import ast
import csv
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from torch import nn
import torch.nn.functional as functional
from monai import transforms
from monai.data.utils import dense_patch_slices
from transformers import BertTokenizer

import inference_demo as demo
from dynamic_network_architectures.vision_branch import VisionBranch
from generate_text_embeddings import merlin_prompts


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = ROOT / "training_results/gradcache48/20261102_201102_310236/checkpoint_9.pth"
DEFAULT_MERLIN_SOURCE = ROOT / "RADAR_train/infer_merlin_anatomy.py"


def organ_mapping():
    source = ROOT / "RADAR_train/lavis/models/radar_models/radar_pretrain.py"
    for node in ast.walk(ast.parse(source.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "organ_dict" for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise ValueError("Cannot find the training model's organ-name mapping")


class CheckpointInference(nn.Module):
    forward_test_win = demo.RADAR.forward_test_win

    def __init__(self):
        super().__init__()
        self.visual_encoder = VisionBranch()
        names = organ_mapping()
        self.organs = [names[organ] for organ in self.visual_encoder.organs]
        self.attention = nn.MultiheadAttention(256, 4, dropout=0.1, batch_first=True)
        self.vision_projs = nn.ModuleList([nn.Linear(256, 256) for organ in self.organs])
        self.query_tokens = nn.Parameter(torch.zeros(len(self.organs), 256))
        self.temp = nn.Parameter(torch.ones([]))


def load_model(checkpoint_path, bert_dir):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
    state = {key.removeprefix("module."): value for key, value in checkpoint["model"].items()}
    tokenizer = BertTokenizer.from_pretrained(str(bert_dir), local_files_only=True)
    word_embeddings = state["text_encoder.embeddings.word_embeddings.weight"]
    if word_embeddings.shape != (len(tokenizer), 768):
        raise ValueError("Checkpoint text vocabulary does not match the training tokenizer")
    if tuple(state["text_proj.weight"].shape) != (256, 768):
        raise ValueError("Checkpoint text projection is incompatible")
    model = CheckpointInference()
    required = model.state_dict()
    selected = {key: value for key, value in state.items() if key in required}
    model.load_state_dict(selected, strict=True)
    if not torch.isfinite(model.temp) or model.temp <= 0:
        raise ValueError("Checkpoint temperature must be finite and positive")
    summary = {
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_epoch_zero_based": checkpoint.get("epoch"),
        "inference_weight_tensors": len(required),
        "text_vocabulary_size": len(tokenizer),
        "temperature": model.temp.item(),
    }
    print(json.dumps(summary, indent=2), flush=True)
    return model.eval(), summary


def load_text_features(args, items, summary):
    prompts = merlin_prompts(args.merlin_source)
    if not args.text_embeddings.exists():
        exporter = Path(__file__).with_name("generate_text_embeddings.py")
        environment = dict(os.environ, OMP_NUM_THREADS=str(args.cpu_threads), MKL_NUM_THREADS=str(args.cpu_threads))
        print(f"Generating {len(prompts)} checkpoint-matched MERLIN text items on CPU.", flush=True)
        subprocess.run([
            sys.executable, str(exporter), "--checkpoint", str(args.checkpoint),
            "--prompt-source", "merlin", "--merlin-source", str(args.merlin_source),
            "--aggregation", "mean", "--max-length", "100", "--output", str(args.text_embeddings),
            "--bert-dir", str(args.bert_dir), "--device", "cpu",
        ], check=True, env=environment)
    metadata_path = args.text_embeddings.with_suffix(args.text_embeddings.suffix + ".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if Path(metadata["checkpoint"]).resolve() != args.checkpoint.resolve():
        raise ValueError("Text embeddings belong to another checkpoint; choose a new output filename")
    if metadata.get("checkpoint_epoch_zero_based") != summary["checkpoint_epoch_zero_based"]:
        raise ValueError("Text embedding epoch differs from checkpoint epoch")
    vocab_hash = hashlib.sha256((args.bert_dir / "vocab.txt").read_bytes()).hexdigest()
    if metadata.get("vocab_sha256") != vocab_hash:
        raise ValueError("Text embeddings use a different tokenizer vocabulary")
    if metadata.get("class_order") != ["negative", "positive"]:
        raise ValueError("Text embeddings must use negative/positive row order")
    if metadata.get("prompt_source") != "merlin" or metadata.get("aggregation") != "mean" or metadata.get("max_length") != 100:
        raise ValueError("Expected MERLIN prompts with max_length=100 and unnormalized class means")
    if metadata.get("prompts") != prompts:
        raise ValueError("Text embedding prompts differ from the selected MERLIN source; regenerate the features")
    features = torch.load(args.text_embeddings, map_location="cpu", weights_only=True)
    if set(features) != set(items):
        raise ValueError("Text embeddings must match the selected MERLIN organ/disease tuple keys")
    for key, value in features.items():
        if not isinstance(value, torch.Tensor) or value.shape != (2, 256) or not torch.isfinite(value).all():
            raise ValueError(f"Invalid text embedding: {key}")
    print(f"Validated {len(features)} checkpoint-matched text items: {args.text_embeddings}", flush=True)
    return features


class MerlinDataFolder(torch.utils.data.Dataset):
    def __init__(self, image_dir, prompt_source):
        self.img_paths = sorted(str(path) for path in image_dir.iterdir() if path.is_file() and path.name.endswith((".nii", ".nii.gz")))
        self.test_items = [tuple(key.split("_", 1)) for key in merlin_prompts(prompt_source)]
        self.test_organs = sorted({item[0] for item in self.test_items})
        self.pad_func = transforms.SpatialPadd(
            keys=["image"], spatial_size=(96, 256, 384), mode="constant", constant_values=0, method="end",
        )

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        image_path = self.img_paths[index]
        loaded = transforms.LoadImaged(keys=["image"], image_only=False, ensure_channel_first=True)({"image": image_path})
        if loaded["image"].ndim != 4 or loaded["image"].shape[0] != 1:
            raise ValueError(f"Expected a single-channel 3D CT: {image_path}")
        affine = loaded["image_meta_dict"]["affine"]
        spacing = [abs(affine[axis, axis].item()) for axis in range(3)]
        height, width, depth = loaded["image"].shape[1:]
        target = [max(1, int(height * spacing[1])), max(1, int(width * spacing[0])), max(1, int(depth * spacing[2] / 5.0))]
        resized = transforms.Compose([
            transforms.Resized(keys=["image"], spatial_size=target, mode="trilinear"),
            transforms.Transposed(keys=["image"], indices=(0, 3, 2, 1)),
        ])(loaded)
        image = resized["image"].as_tensor().clamp(-1000, 1000)
        if not torch.isfinite(image).all():
            raise ValueError(f"Non-finite CT intensities: {image_path}")
        intensity_range = image.max() - image.min()
        if intensity_range <= 0:
            raise ValueError(f"Constant CT intensities: {image_path}")
        image = (image - image.min()) / intensity_range
        foreground = image[0].ne(0)
        bounds = []
        for axis, margin in enumerate((5, 20, 20)):
            occupied = foreground.any(dim=tuple(other for other in range(3) if other != axis)).nonzero().flatten()
            start = max(0, occupied[0].item() - margin)
            end = min(image.shape[axis + 1], occupied[-1].item() + margin)
            bounds.append(slice(start, end))
        image = self.pad_func({"image": image[(slice(None), *bounds)]})["image"]
        metadata = {
            "file_name": Path(image_path).name, "img_path": image_path,
            "test_organ_names": self.test_organs,
        }
        return image, self.test_items, metadata


def load_dataset(image_dir, prompt_source=DEFAULT_MERLIN_SOURCE, report_json=None):
    dataset = MerlinDataFolder(image_dir, prompt_source)
    if report_json is not None:
        reports = json.loads(report_json.read_text(encoding="utf-8-sig"))
        filenames = sorted(f"{patient_id}.nii.gz" for patient_id, record in reports.items() if record.get("split") == "test")
        missing = [filename for filename in filenames if not (image_dir / filename).is_file()]
        if missing:
            raise FileNotFoundError(f"Missing {len(missing)} MERLIN test images, including {missing[:5]}")
        dataset.img_paths = [str(image_dir / filename) for filename in filenames]
        print(f"Selected {len(filenames)} MERLIN test images from {report_json}", flush=True)
    if not dataset.img_paths:
        raise ValueError(f"No .nii/.nii.gz files found in {image_dir}")
    return dataset


@torch.inference_mode()
def infer_case(model, image, items, metadata, text_features, pad, device, sw_batch_size=1, segmentation_device="auto"):
    image = image.unsqueeze(0)
    image_size = list(image.shape[2:])
    roi_size = (96, 256, 384)
    interval = demo._get_scan_interval(image_size, roi_size, 3, 0.25)
    windows = dense_patch_slices(image_size, roi_size, interval)
    organ_logits = {item: [] for item in items}
    organ_features = {}
    accumulation_device = torch.device("cpu")
    mask_bytes = 37 * math.prod(image_size) * 4
    if device.type == "cuda" and segmentation_device != "cpu":
        free_memory, _ = torch.cuda.mem_get_info(device)
        free_memory += max(0, torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device))
        budget = min(free_memory // 4, max(0, free_memory - 8 * 1024**3))
        if segmentation_device == "cuda" or mask_bytes <= budget:
            accumulation_device = device
    if segmentation_device == "cuda" and device.type != "cuda":
        raise ValueError("CUDA segmentation accumulation requires a CUDA inference device")
    try:
        full_mask = torch.zeros((1, 37, *image_size), dtype=torch.float32, device=accumulation_device)
    except torch.cuda.OutOfMemoryError:
        if segmentation_device != "auto":
            raise
        accumulation_device = torch.device("cpu")
        full_mask = torch.zeros((1, 37, *image_size), dtype=torch.float32)
    print(f"  segmentation accumulation: {accumulation_device}, buffer GiB={mask_bytes / 1024**3:.2f}", flush=True)
    for window_start in range(0, len(windows), sw_batch_size):
        batch_windows = windows[window_start:window_start + sw_batch_size]
        selections = [(slice(None), slice(None), *window) for window in batch_windows]
        patch = torch.cat([image[selection] for selection in selections], dim=0).to(device)
        organ_logits, probabilities = model.forward_test_win(
            patch, None, organ_logits, metadata["test_organ_names"], text_features, organ_features, None,
        )
        probabilities = functional.interpolate(probabilities, size=patch.shape[2:], mode="trilinear")
        probabilities = probabilities.to(device=accumulation_device, dtype=torch.float32)
        for batch_index, selection in enumerate(selections):
            full_mask[selection] += probabilities[batch_index:batch_index + 1]
        print(f"  windows {window_start + 1}-{window_start + len(batch_windows)}/{len(windows)}", flush=True)
        del patch, probabilities
    segmentation = full_mask.argmax(1).to(dtype=torch.uint8).cpu().unsqueeze(0)
    del full_mask
    missing_organs = sorted({item[0] for item, values in organ_logits.items() if not values})
    for organ in missing_organs:
        if organ in organ_features:
            continue
        organ_id = model.organs.index(organ)
        organ_mask = segmentation.eq(organ_id + 1)
        if not organ_mask.any():
            continue
        patch, patch_mask = demo.center_crop(image, organ_mask, roi_size)
        padded = pad({"image": patch[0], "label": patch_mask[0].float()})
        organ_logits, _ = model.forward_test_win(
            padded["image"].unsqueeze(0).to(device), None, organ_logits,
            metadata["test_organ_names"], text_features, organ_features, None, skip_organ=organ_id,
        )
    scores = [np.concatenate(organ_logits[item]).mean(0)[1].item() if organ_logits[item] else None for item in items]
    unscored = {
        item[0]: "no_valid_organ_feature" if segmentation.eq(model.organs.index(item[0]) + 1).any() else "organ_not_detected_or_outside_scan"
        for item, values in organ_logits.items() if not values
    }
    return [metadata["file_name"], *scores], unscored


def parse_args(defaults=None):
    parser = argparse.ArgumentParser(description="Checkpoint-matched MERLIN disease inference on YG NIfTI images")
    parser.add_argument("--checkpoint", type=Path, default=Path(os.environ.get("CHECKPOINT_PATH", str(DEFAULT_CHECKPOINT))))
    parser.add_argument("--text-embeddings", type=Path, default=Path(os.environ.get("TEXT_EMBEDDING_PATH", str(ROOT / "ckpt/infer_text_embedding_epoch10_merlin_anatomy.pt"))))
    parser.add_argument("--bert-dir", type=Path, default=ROOT / "ckpt/bert-base-uncased")
    parser.add_argument("--merlin-source", type=Path, default=DEFAULT_MERLIN_SOURCE)
    parser.add_argument("--img-dir", "--img_dir", dest="img_dir", type=Path, default=Path("/data/dataset/yg_images"))
    parser.add_argument("--report-json", type=Path, help="Select only split=test IDs from this MERLIN report JSON")
    parser.add_argument("--save-dir", "--save_dir", dest="save_dir", type=Path, default=ROOT / "results/yg_images")
    parser.add_argument("--save-tag", "--save_tag", dest="save_tag", default="epoch10_merlin")
    device_options = parser.add_mutually_exclusive_group()
    device_options.add_argument("--device", default="cuda:0", help="Single device, e.g. cuda:0 or cpu")
    device_options.add_argument("--devices", help="Comma-separated CUDA indices visible to this process, e.g. 0,1,2,3")
    parser.add_argument("--shard-index", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--num-shards", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--sw-batch-size", type=int, default=1, help="Sliding windows per GPU forward pass; increase with available GPU memory")
    parser.add_argument("--segmentation-device", choices=("auto", "cpu", "cuda"), default="auto", help="Accumulate segmentation on GPU when memory permits; cpu restores host accumulation")
    parser.add_argument("--workers-per-gpu", type=int, default=1, help="Independent case processes per selected GPU; requires --devices")
    parser.add_argument("--no-prefetch", action="store_true", help="Disable overlapping next-image loading with inference")
    parser.add_argument("--limit", type=int, help="Optional maximum number of images")
    parser.add_argument("--check-only", action="store_true", help="Validate real checkpoint, text features and input headers on CPU")
    if defaults:
        parser.set_defaults(**defaults)
    args = parser.parse_args()
    if args.cpu_threads < 1 or (args.limit is not None and args.limit < 1):
        parser.error("cpu-threads and limit must be positive")
    if args.sw_batch_size < 1:
        parser.error("sw-batch-size must be positive")
    if args.workers_per_gpu < 1:
        parser.error("workers-per-gpu must be positive")
    if args.workers_per_gpu > 1 and args.devices is None:
        parser.error("Use --devices 0 --workers-per-gpu 2 for two processes on one GPU")
    if args.sw_batch_size > 1 and not args.no_prefetch:
        print("Disabling threaded prefetch with batched windows: this runtime requires synchronous image loading for validated batched results.", flush=True)
        args.no_prefetch = True
    if Path(args.save_tag).name != args.save_tag or args.save_tag in ("", ".", ".."):
        parser.error("save-tag must be a filename tag")
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("Invalid worker shard index/count")
    if args.devices is not None:
        tokens = [token.strip() for token in args.devices.split(",")]
        if any(not token.isdecimal() for token in tokens):
            parser.error("--devices must contain CUDA indices, e.g. 0,1,2,3")
        args.devices = [int(token) for token in tokens]
        if len(set(args.devices)) != len(args.devices):
            parser.error("--devices must not contain duplicate indices")
        if args.num_shards != 1:
            parser.error("Workers cannot launch additional workers")
    return args


def merge_shards(shard_outputs, expected_paths, output, devices):
    expected = [Path(path).name for path in expected_paths]
    if len(set(expected)) != len(expected):
        raise ValueError("Selected image filenames are not unique")
    rows = {}
    cases = {}
    columns = None
    summary = None
    for shard_index, shard_output in enumerate(shard_outputs):
        with shard_output.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if columns is None:
                columns = reader.fieldnames
            elif reader.fieldnames != columns:
                raise ValueError("Worker output columns differ")
            shard_rows = list(reader)
        metadata = json.loads(shard_output.with_suffix(".json").read_text(encoding="utf-8"))
        if summary is None:
            summary = {key: value for key, value in metadata.items() if key not in ("cases", "shard_index", "num_shards", "processed_images", "device")}
        elif any(metadata.get(key) != value for key, value in summary.items()):
            raise ValueError("Worker inference settings differ")
        shard_names = [row["file_name"] for row in shard_rows]
        if shard_names != expected[shard_index::len(shard_outputs)]:
            raise ValueError(f"Worker {shard_index} output is incomplete or in the wrong order")
        if set(metadata["cases"]) != set(shard_names):
            raise ValueError(f"Worker {shard_index} case metadata differs from its CSV")
        for row in shard_rows:
            filename = row["file_name"]
            if filename in rows:
                raise ValueError(f"Duplicate worker result: {filename}")
            for key, value in row.items():
                if key != "file_name" and value and not 0 <= float(value) <= 1:
                    raise ValueError(f"Invalid worker score: {filename}, {key}")
            rows[filename] = row
        cases.update(metadata["cases"])
    if set(rows) != set(expected) or not columns or summary is None:
        raise ValueError("Merged results contain missing or unexpected cases")
    temporary_csv = output.with_suffix(".csv.tmp")
    temporary_json = output.with_suffix(".json.tmp")
    with temporary_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows[filename] for filename in expected)
    temporary_json.write_text(json.dumps({
        **summary, "devices": devices, "num_workers": len(shard_outputs),
        "processed_images": len(rows), "cases": {filename: cases[filename] for filename in expected},
        "worker_outputs": [str(path.resolve()) for path in shard_outputs],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary_csv.replace(output)
    temporary_json.replace(output.with_suffix(".json"))
    print(f"Merged {len(rows)} unique cases from {len(shard_outputs)} workers into {output}", flush=True)
    return output


def run_workers(args, selected_paths):
    devices = (args.devices * args.workers_per_gpu)[:len(selected_paths)]
    if not args.check_only:
        available = torch.cuda.device_count()
        if any(device >= available for device in devices):
            raise ValueError(f"Requested CUDA devices {devices}, but only {available} devices are visible")
    args.save_dir.mkdir(parents=True, exist_ok=True)
    worker_dir = Path(tempfile.mkdtemp(prefix=f"{args.save_tag}_workers_", dir=args.save_dir))
    processes = []
    handles = []
    outputs = []
    print(f"Distributing {len(selected_paths)} cases across devices {devices}; worker logs: {worker_dir}", flush=True)
    try:
        for shard_index, device_index in enumerate(devices):
            tag = f"shard_{shard_index:03d}"
            command = [
                sys.executable, "-u", str(Path(__file__).resolve()),
                "--checkpoint", str(args.checkpoint.resolve()),
                "--text-embeddings", str(args.text_embeddings.resolve()),
                "--bert-dir", str(args.bert_dir.resolve()), "--merlin-source", str(args.merlin_source.resolve()),
                "--img-dir", str(args.img_dir.resolve()), "--save-dir", str(worker_dir.resolve()),
                "--save-tag", tag, "--device", f"cuda:{device_index}",
                "--cpu-threads", str(args.cpu_threads), "--num-shards", str(len(devices)),
                "--sw-batch-size", str(args.sw_batch_size),
                "--segmentation-device", args.segmentation_device,
                "--shard-index", str(shard_index),
            ]
            if args.report_json:
                command.extend(["--report-json", str(args.report_json.resolve())])
            if args.limit is not None:
                command.extend(["--limit", str(args.limit)])
            if args.check_only:
                command.append("--check-only")
            if args.no_prefetch:
                command.append("--no-prefetch")
            log_path = worker_dir / f"{tag}.log"
            handle = log_path.open("w", encoding="utf-8")
            handles.append(handle)
            environment = dict(os.environ, OMP_NUM_THREADS=str(args.cpu_threads), MKL_NUM_THREADS=str(args.cpu_threads))
            processes.append(subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, env=environment))
            outputs.append(worker_dir / f"RADAR_infer_results_{tag}.csv")
            print(f"Worker {shard_index}: cuda:{device_index}, {len(selected_paths[shard_index::len(devices)])} cases", flush=True)
        pending = set(range(len(processes)))
        while pending:
            for shard_index in list(pending):
                code = processes[shard_index].poll()
                if code is None:
                    continue
                if code != 0:
                    log_path = worker_dir / f"shard_{shard_index:03d}.log"
                    tail = log_path.read_text(encoding="utf-8", errors="replace")[-6000:]
                    raise RuntimeError(f"Worker {shard_index} failed with exit code {code}; see {log_path}\n{tail}")
                pending.remove(shard_index)
                print(f"Worker {shard_index} completed", flush=True)
            if pending:
                time.sleep(1)
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for handle in handles:
            handle.close()
    if args.check_only:
        print(f"Multi-worker compatibility checks passed for {len(selected_paths)} cases; no GPU allocated.", flush=True)
        return None
    output = args.save_dir / f"RADAR_infer_results_{args.save_tag}.csv"
    return merge_shards(outputs, selected_paths, output, devices)


def main(defaults=None):
    args = parse_args(defaults)
    torch.set_num_threads(args.cpu_threads)
    dataset = load_dataset(args.img_dir, args.merlin_source, args.report_json)
    model, summary = load_model(args.checkpoint, args.bert_dir)
    if set(dataset.test_organs) - set(model.organs):
        raise ValueError("MERLIN prompt regions do not match the training organ mapping")
    text_features = load_text_features(args, dataset.test_items, summary)
    selected_paths = dataset.img_paths[:args.limit] if args.limit is not None else dataset.img_paths
    if args.devices is not None:
        del model, text_features
        gc.collect()
        return run_workers(args, selected_paths)
    dataset.img_paths = selected_paths[args.shard_index::args.num_shards]
    summary.update({
        "protocol": "repository_merlin_anatomy", "disease_items": len(dataset.test_items),
        "prompt_source": str(args.merlin_source.resolve()), "aggregation": "mean",
        "clip_hu": [-1000, 1000], "reference_spacing": [1.0, 1.0, 5.0],
        "ground_truth_available": False, "evaluation_metrics": None,
        "score_type": "negative_positive_prompt_softmax",
        "report_json": str(args.report_json.resolve()) if args.report_json else None,
        "split": "test" if args.report_json else None,
        "selected_images": len(selected_paths),
        "shard_index": args.shard_index, "num_shards": args.num_shards,
        "device": args.device,
        "sw_batch_size": args.sw_batch_size,
        "segmentation_device": args.segmentation_device,
    })
    count = len(dataset)
    if args.check_only:
        for image_path in dataset.img_paths:
            reader = demo.sitk.ImageFileReader()
            reader.SetFileName(image_path)
            reader.ReadImageInformation()
            if reader.GetDimension() != 3:
                raise ValueError(f"Expected a 3D CT: {image_path}")
            print(f"Input: {Path(image_path).name}, shape={reader.GetSize()}, spacing={reader.GetSpacing()}", flush=True)
        print(f"Compatibility checks passed: {len(dataset)} selected images, {count} headers checked; no GPU allocated.", flush=True)
        return
    device = torch.device(args.device)
    model.to(device)
    text_features = {key: value.to(device=device, dtype=torch.float32) for key, value in text_features.items()}
    pad = transforms.DivisiblePadd(keys=["image", "label"], k=32, mode="constant", constant_values=0, method="end")
    args.save_dir.mkdir(parents=True, exist_ok=True)
    output = args.save_dir / f"RADAR_infer_results_{args.save_tag}.csv"
    columns = ["file_name", *["_".join(item) for item in dataset.test_items]]
    statuses = {}
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with ThreadPoolExecutor(max_workers=1) as loader, output.open("w", encoding="utf-8-sig", newline="") as csv_handle, output.with_suffix(".cases.jsonl").open("w", encoding="utf-8") as journal:
        writer = csv.writer(csv_handle)
        writer.writerow(columns)
        future = loader.submit(dataset.__getitem__, 0) if count and not args.no_prefetch else None
        for image_index in range(count):
            case_started = time.perf_counter()
            image, items, metadata = future.result() if future is not None else dataset[image_index]
            future = loader.submit(dataset.__getitem__, image_index + 1) if image_index + 1 < count and not args.no_prefetch else None
            print(f"Case {image_index + 1}/{count}: {metadata['file_name']}", flush=True)
            row, unscored = infer_case(model, image, items, metadata, text_features, pad, device, args.sw_batch_size, args.segmentation_device)
            statuses[metadata["file_name"]] = {"unscored_organs": unscored}
            writer.writerow(row)
            csv_handle.flush()
            journal.write(json.dumps({"file_name": metadata["file_name"], "unscored_organs": unscored}, ensure_ascii=False) + "\n")
            journal.flush()
            elapsed = time.perf_counter() - started
            print(f"Case seconds={time.perf_counter() - case_started:.2f}, elapsed={elapsed:.2f}, ETA seconds={elapsed / (image_index + 1) * (count - image_index - 1):.2f}", flush=True)
            del image
    peak_memory = torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else 0
    output.with_suffix(".json").write_text(json.dumps({**summary, "text_embeddings": str(args.text_embeddings.resolve()), "processed_images": count, "cases": statuses}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {count} cases to {output}; elapsed seconds={time.perf_counter() - started:.2f}; peak GPU allocated GiB={peak_memory:.2f}", flush=True)
    return output


if __name__ == "__main__":
    main()
