import argparse
from collections import defaultdict, deque
from datetime import datetime
import json
from importlib.metadata import version
import logging
import math
import os
from pathlib import Path
import random
import resource
import time

import numpy as np
from omegaconf import OmegaConf
import torch
import torch.distributed as distributed
from torch.utils.data import DataLoader, Subset

from gradient_cache import GradientCache, gather_objects, synchronize_gradients, world_size
from lavis.datasets.datasets.caption_datasets import CaptionDataset
from lavis.models.radar_models.radar_pretrain import RadarPretrain
from lavis.processors.radar_processors import RadarImageTrainProcessor, RadarReportProcessor


def save_checkpoint(path, model, optimizer, scaler, config, epoch, step):
    random_states = gather_objects({
        "cpu_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(),
        "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
    })
    if distributed.is_initialized() and distributed.get_rank() != 0:
        return
    temporary = path.with_suffix(".tmp")
    torch.save({
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(), "config": OmegaConf.to_container(config),
        "epoch": epoch, "global_step": step,
        "cpu_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(),
        "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
        "trainer": "gradient_cache_v1",
        "rank_rng": random_states, "world_size": world_size(),
    }, temporary)
    temporary.replace(path)


def learning_rate(config, epoch, step):
    if step < config.run.warmup_steps:
        fraction = step / max(config.run.warmup_steps, 1)
        return config.run.warmup_lr + fraction * (config.run.init_lr - config.run.warmup_lr)
    return config.run.min_lr + 0.5 * (config.run.init_lr - config.run.min_lr) * (
        1 + math.cos(math.pi * epoch / config.run.max_epoch)
    )


def main():
    parser = argparse.ArgumentParser(description="RADAR distributed contrastive gradient cache")
    parser.add_argument("--cfg-path", default="radar_gradcache.yaml")
    parser.add_argument("--options", nargs="*", default=[])
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--verify-replay", action="store_true")
    args = parser.parse_args()
    config = OmegaConf.merge(OmegaConf.load(args.cfg_path), OmegaConf.from_dotlist(args.options))
    os.environ["TZ"] = "Asia/Shanghai"
    time.tzset()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        distributed.init_process_group(backend="nccl", device_id=torch.device("cuda", local_rank))
    rank = distributed.get_rank() if distributed.is_initialized() else 0
    if config.run.accum_grad_iters != 1:
        raise ValueError("Gradient cache performs one update per contrastive batch; accum_grad_iters must be 1.")
    micro_batch = int(config.run.batch_size_train)
    contrastive_batch = int(config.run.contrastive_batch_size)
    if micro_batch < 1 or contrastive_batch < 2 or contrastive_batch % (micro_batch * world_size()):
        raise ValueError("contrastive_batch_size must be >=2 and divisible by batch_size_train * WORLD_SIZE")
    local_cases = contrastive_batch // world_size()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    resource.setrlimit(resource.RLIMIT_CORE, (0, resource.getrlimit(resource.RLIMIT_CORE)[1]))
    seed = int(config.run.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.autograd.set_detect_anomaly(False)
    precision = config.run.precision
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("BF16 is not supported by this GPU")
    output = Path(config.run.output_dir) / gather_objects(datetime.now().strftime("%Y%m%d_%H%M%S_%f"))[0]
    output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", handlers=[
        logging.StreamHandler() if rank == 0 else logging.NullHandler(),
        logging.FileHandler(output / ("train.log" if rank == 0 else f"rank_{rank}.log")),
    ])
    if rank == 0:
        OmegaConf.save(config, output / "config.yaml")
    environment = {name: version(name) for name in (
        "torch", "torchvision", "numpy", "transformers", "monai", "timm", "nibabel", "SimpleITK",
    )}
    environment.update({"world_size": world_size(), "contrastive_batch_size": contrastive_batch})
    if rank == 0:
        (output / "environment.json").write_text(json.dumps(environment, indent=2))
    logging.info("output=%s torch=%s cuda=%s GPU=%s", output, torch.__version__, torch.version.cuda, torch.cuda.get_device_name())
    dataset = CaptionDataset(RadarImageTrainProcessor(), RadarReportProcessor(), "", [])
    dataset.strict_loading = True
    if config.get("data"):
        dataset.annotation = json.loads(Path(config.data.reports).read_text())
        dataset.organ_abnormal_info = json.loads(Path(config.data.abnormal).read_text())
        dataset.patient_paths = sorted(str(path) for path in Path(config.data.images).glob("*.nii.gz"))
    if len(dataset) < contrastive_batch:
        raise ValueError(f"Need at least {contrastive_batch} training cases; found {len(dataset)}")
    if len(set(dataset.patient_paths)) != len(dataset.patient_paths):
        raise ValueError("Duplicate training image paths")
    for path in dataset.patient_paths:
        patient = Path(path).name.removesuffix(".nii.gz")
        if patient not in dataset.annotation or patient not in dataset.organ_abnormal_info:
            raise ValueError(f"Missing report/abnormal annotation: {patient}")
        if not Path(path.replace("resized_images", "resized_masks")).is_file():
            raise ValueError(f"Missing mask: {path}")
    model = RadarPretrain.from_config(config.model).cuda().train()
    if world_size() > 1:
        with torch.no_grad():
            for tensor in list(model.parameters()) + list(model.buffers()):
                distributed.broadcast(tensor, src=0)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(config.run.init_lr), weight_decay=float(config.run.weight_decay),
        betas=(0.9, float(config.run.get("beta2", 0.999))),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")
    cache = GradientCache(model, precision=precision, verify_replay=args.verify_replay)
    start_epoch, global_step = 0, 0
    resume = config.run.get("resume_ckpt_path")
    if resume:
        checkpoint = torch.load(resume, map_location="cpu", weights_only=False)
        if checkpoint.get("trainer") != "gradient_cache_v1":
            raise ValueError("Resume requires a gradient-cache checkpoint. Use init_checkpoint for old model weights.")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch, global_step = checkpoint["epoch"] + 1, checkpoint["global_step"]
        states = checkpoint.get("rank_rng", [])
        if checkpoint.get("world_size", 1) == world_size() and len(states) == world_size():
            state = states[rank]
        elif checkpoint.get("world_size", 1) == 1 and world_size() == 1:
            state = checkpoint
        else:
            state = None
            logging.info("GPU count changed; optimizer and schedule restored, random streams reseeded")
        if state is not None:
            torch.set_rng_state(state["cpu_rng"])
            torch.cuda.set_rng_state(state["cuda_rng"])
            random.setstate(state["python_rng"])
            np.random.set_state(state["numpy_rng"])
        logging.info("Resumed completed epoch %d at update %d", checkpoint["epoch"], global_step)
    elif config.run.get("init_checkpoint"):
        checkpoint = torch.load(config.run.init_checkpoint, map_location="cpu", weights_only=False)
        message = model.load_state_dict(checkpoint["model"], strict=False)
        model.copy_params()
        logging.info("Initialized model weights only: %s", message)
    if not resume or checkpoint.get("world_size", 1) != world_size():
        random.seed(seed + rank)
        np.random.seed(seed + rank)
        torch.manual_seed(seed + rank)
        torch.cuda.manual_seed(seed + rank)
    updates_per_epoch = len(dataset) // contrastive_batch
    logging.info("cases=%d micro_batch=%d contrastive_batch=%d updates_per_epoch=%d remainder_dropped=%d",
                 len(dataset), micro_batch, contrastive_batch, updates_per_epoch, len(dataset) % contrastive_batch)
    logging.info("world_size=%d local_cases=%d microbatches_per_rank=%d", world_size(), local_cases, local_cases // micro_batch)
    windows = defaultdict(lambda: deque(maxlen=100))
    skipped_updates = 0
    training_started = time.monotonic()
    journal_path = output / "metrics.jsonl" if rank == 0 else Path(os.devnull)
    with journal_path.open("a", buffering=1) as journal:
        for epoch in range(start_epoch, int(config.run.max_epoch)):
            generator = torch.Generator().manual_seed(seed + epoch)
            order = torch.randperm(len(dataset), generator=generator).tolist()
            indices = order[rank:updates_per_epoch * contrastive_batch:world_size()]
            loader = DataLoader(
                Subset(dataset, indices), batch_size=micro_batch, shuffle=False, generator=generator,
                num_workers=int(config.run.num_workers), pin_memory=False, drop_last=True,
            )
            iterator = iter(loader)
            epoch_losses = defaultdict(list)
            for update in range(updates_per_epoch):
                started = time.monotonic()
                samples = [next(iterator) for _ in range(local_cases // micro_batch)]
                ids = sum([list(sample["patient_id"]) for sample in samples], [])
                ids = sum(gather_objects(ids), [])
                if len(set(ids)) != contrastive_batch:
                    raise ValueError("A contrastive batch contains duplicate patients, possibly due to dataset retry logic")
                load_seconds = time.monotonic() - started
                branch = "global" if global_step % 2 == 0 and model.radar_plus else "organ"
                rate = learning_rate(config, epoch, global_step)
                for group in optimizer.param_groups:
                    group["lr"] = rate
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.reset_peak_memory_stats()
                metrics = cache.backward(samples, branch, scaler)
                synchronize_gradients(model)
                scaler.unscale_(optimizer)
                gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
                scale_before = scaler.get_scale()
                if not torch.isfinite(gradient_norm) and precision != "fp16":
                    raise FloatingPointError(f"Non-finite gradients at update {global_step}")
                scaler.step(optimizer)
                scaler.update()
                skipped = scaler.get_scale() < scale_before
                skipped_updates += int(skipped)
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                metrics.update({
                    "world_size": world_size(), "cases_per_rank": local_cases,
                    "epoch": epoch, "update": update, "global_step": global_step,
                    "lr": rate, "gradient_norm": gradient_norm.item() if torch.isfinite(gradient_norm) else None,
                    "amp_scale": scaler.get_scale(), "update_skipped": skipped,
                    "skipped_updates": skipped_updates, "load_seconds": load_seconds,
                    "step_seconds": time.monotonic() - started,
                    "allocated_mib": torch.cuda.memory_allocated() / 2**20,
                    "reserved_mib": torch.cuda.memory_reserved() / 2**20,
                    "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                    "patient_ids": ids, "timestamp": datetime.now().isoformat(),
                })
                windows[branch].append(metrics["loss_itc"])
                metrics["loss_itc_mean_100"] = sum(windows[branch]) / len(windows[branch])
                epoch_losses[branch].append(metrics["loss_itc"])
                if not all(math.isfinite(metrics[key]) for key in ("loss", "loss_itc", "loss_seg")):
                    raise FloatingPointError(f"Non-finite loss at update {global_step}")
                journal.write(json.dumps(metrics, ensure_ascii=False, allow_nan=False) + "\n")
                if update % int(config.run.log_freq) == 0:
                    logging.info("epoch=%d update=%d/%d branch=%s cases=%d pairs=%d itc=%.5f mean100=%.5f seg=%.5f grad=%s lr=%.3g seconds=%.2f peak_MiB=%.0f skipped=%s",
                                 epoch, update, updates_per_epoch, branch, metrics["cases"], metrics["pairs"],
                                 metrics["loss_itc"], metrics["loss_itc_mean_100"], metrics["loss_seg"],
                                 metrics["gradient_norm"], rate, metrics["step_seconds"], metrics["peak_allocated_mib"], skipped)
                global_step += 1
                if args.max_steps and global_step >= args.max_steps:
                    logging.info("Smoke test complete: %d updates, no production training started", global_step)
                    return
                del samples
            save_checkpoint(output / f"checkpoint_{epoch}.pth", model, optimizer, scaler, config, epoch, global_step)
            summary = {branch: sum(values) / len(values) for branch, values in epoch_losses.items()}
            logging.info("epoch=%d branch_mean_itc=%s elapsed_hours=%.3f checkpoint_saved=true", epoch, summary, (time.monotonic() - training_started) / 3600)


if __name__ == "__main__":
    try:
        main()
    finally:
        if distributed.is_initialized():
            distributed.destroy_process_group()
