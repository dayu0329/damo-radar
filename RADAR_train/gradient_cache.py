import contextlib
from dataclasses import dataclass
import logging
import time

import torch
import torch.distributed as distributed
import torch.nn.functional as functional


@dataclass
class RandomState:
    cpu: torch.Tensor
    cuda: torch.Tensor

    @classmethod
    def capture(cls):
        return cls(torch.get_rng_state(), torch.cuda.get_rng_state())

    @contextlib.contextmanager
    def replay(self):
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.set_rng_state(self.cpu)
            torch.cuda.set_rng_state(self.cuda)
            yield


def snapshot_buffers(model):
    return {name: buffer.clone() for name, buffer in model.named_buffers()}


@torch.no_grad()
def restore_buffers(model, saved):
    for name, buffer in model.named_buffers():
        buffer.copy_(saved[name])


def world_size():
    return distributed.get_world_size() if distributed.is_initialized() else 1


def gather_objects(value):
    if world_size() == 1:
        return [value]
    gathered = [None] * world_size()
    distributed.all_gather_object(gathered, value)
    return gathered


def gather_groups(groups):
    if world_size() == 1:
        return groups
    payload = {
        key: [{name: value.cpu() if isinstance(value, torch.Tensor) else value
               for name, value in entry.items() if name in ("image", "text", "teacher", "abnormal", "captions")}
              for entry in entries]
        for key, entries in groups.items()
    }
    combined = {}
    for rank, remote in enumerate(gather_objects(payload)):
        for key in sorted(remote):
            entries = groups[key] if rank == distributed.get_rank() else [
                {name: value.cuda() if isinstance(value, torch.Tensor) else value
                 for name, value in entry.items()} for entry in remote[key]
            ]
            combined.setdefault(key, []).extend(entries)
    return {key: combined[key] for key in sorted(combined)}


@torch.no_grad()
def synchronize_gradients(model):
    if world_size() == 1:
        return
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    used = torch.tensor([parameter.grad is not None for parameter in parameters], device="cuda", dtype=torch.int32)
    distributed.all_reduce(used)
    bucket = []
    elements = 0

    def flush():
        flat = torch.cat([parameter.grad.reshape(-1) for parameter in bucket])
        distributed.all_reduce(flat)
        offset = 0
        for parameter in bucket:
            parameter.grad.copy_(flat[offset:offset + parameter.numel()].view_as(parameter))
            offset += parameter.numel()

    for parameter, active in zip(parameters, used.tolist()):
        if not active:
            continue
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        bucket.append(parameter)
        elements += parameter.numel()
        if elements >= 4 * 1024 * 1024:
            flush()
            bucket = []
            elements = 0
    if bucket:
        flush()


def symmetric_contrastive(image, text, temperature, targets):
    logits = image.float() @ text.float().t() / temperature.float()
    return -0.5 * (
        (functional.log_softmax(logits, dim=1) * targets).sum(1).mean()
        + (functional.log_softmax(logits.t(), dim=1) * targets).sum(1).mean()
    )


def organ_targets(teacher, abnormal, captions, temperature):
    with torch.no_grad():
        normal = ~abnormal
        equal = torch.tensor(
            [[left == right for right in captions] for left in captions],
            device=teacher.device,
            dtype=torch.bool,
        )
        targets = (normal[:, None] & normal[None, :]) | equal
        targets = targets.float()
        semantic = functional.softmax(
            teacher.float() @ teacher.float().t() / temperature.float(), dim=1
        )
        targets += (abnormal[:, None] & abnormal[None, :]).float() * semantic
        targets.fill_diagonal_(1)
        return targets / targets.sum(1, keepdim=True)


def alignment_metrics(image, text, temperature):
    similarity = image.float() @ text.float().t()
    count = len(image)
    diagonal = similarity.diag().mean()
    negative = (similarity.sum() - similarity.diag().sum()) / max(count * (count - 1), 1)
    labels = torch.arange(count, device=image.device)
    return {
        "pairs": count,
        "positive_similarity": diagonal.item(),
        "negative_similarity": negative.item(),
        "similarity_gap": (diagonal - negative).item(),
        "retrieval_top1": 0.5 * (
            (similarity.argmax(1) == labels).float().mean()
            + (similarity.argmax(0) == labels).float().mean()
        ).item(),
        "image_feature_std": image.float().std(dim=0, unbiased=False).mean().item(),
        "text_feature_std": text.float().std(dim=0, unbiased=False).mean().item(),
        "temperature": temperature.item(),
    }


def validate_sample(sample):
    image, label = sample["image"], sample["seg"]
    invalid = (~torch.isfinite(label)) | (label < 0) | (label > 36) | (label != label.round())
    if invalid.any() or not torch.isfinite(image).all():
        raise ValueError(f"Invalid image or segmentation labels: {sample['image_path']}")


def encode_visual(model, sample, branch):
    output = model.visual_encoder(sample["image"], None)
    probabilities, segmentation, *features = output
    embeddings = features[:3]
    token_flags = features[3:]
    if branch == "global":
        keys = torch.cat(embeddings, dim=1)
        query = model.query_tokens_whole.unsqueeze(0).expand(len(keys), -1, -1)
        attended, _ = model.attention_whole(query, keys, keys)
        image = functional.normalize(model.vision_projs_whole(attended.squeeze(1)), dim=-1)
        return {"global": (image, torch.arange(len(image), device=image.device))}, probabilities
    intact = torch.zeros((len(segmentation), len(model.organs)), dtype=torch.bool, device=segmentation.device)
    for patient_index, mask in enumerate(segmentation):
        boundary = torch.unique(torch.cat([
            mask[0].flatten(), mask[-1].flatten(), mask[:, 0].flatten(),
            mask[:, -1].flatten(), mask[:, :, 0].flatten(), mask[:, :, -1].flatten(),
        ]))
        present = torch.unique(mask)
        complete = present[(present > 0) & ~torch.isin(present, boundary)].long()
        intact[patient_index, complete - 1] = True
    result = {}
    for organ_index, organ in enumerate(model.organs):
        patients = torch.where(intact[:, organ_index])[0]
        if len(patients):
            image = model.get_roi_features(*embeddings, *token_flags, patients, organ_index)
            image = functional.normalize(model.vision_projs[organ_index](image), dim=-1)
            result[organ] = (image, patients)
    return result, probabilities


def encode_text(model, tokenized, teacher=False):
    encoder = model.text_encoder_m if teacher else model.text_encoder
    projection = model.text_proj_m if teacher else model.text_proj
    output = encoder.forward_text(tokenized).last_hidden_state[:, 0]
    return functional.normalize(projection(output), dim=-1)


class GradientCache:
    def __init__(self, model, precision="bf16", verify_replay=False, itc_weight=1.0, seg_weight=1.0, organ_text_dynamic_padding=False, organ_text_batch_size=0):
        self.model = model
        self.dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]
        self.verify_replay = verify_replay
        self.itc_weight = float(itc_weight)
        self.seg_weight = float(seg_weight)
        self.organ_text_dynamic_padding = bool(organ_text_dynamic_padding)
        self.organ_text_batch_size = int(organ_text_batch_size)
        if self.organ_text_batch_size < 0:
            raise ValueError("organ_text_batch_size must be >= 0")

    def autocast(self):
        return torch.autocast("cuda", dtype=self.dtype, enabled=self.dtype != torch.float32)

    def text_input(self, captions, organ=False):
        return self.model.tokenizer(
            # 仅器官报告按本批最长文本补齐；截断上限仍由 model.max_txt_len 控制。
            captions, padding="longest" if organ and self.organ_text_dynamic_padding else "max_length", truncation=True,
            max_length=self.model.max_txt_len, return_tensors="pt",
        ).to("cuda")

    def cache_organ_text(self, groups):
        # 编码时跨器官合批；随后按原顺序拆回 entry，器官损失分组保持不变。
        entries = [entry for group in groups.values() for entry in group]
        captions = [caption for entry in entries for caption in entry["captions"]]
        chunks, student_parts, teacher_parts = [], [], []
        token_slots = 0
        for start in range(0, len(captions), self.organ_text_batch_size):
            end = min(start + self.organ_text_batch_size, len(captions))
            tokens = self.text_input(captions[start:end], organ=True)
            state = RandomState.capture()
            with self.autocast():
                text = encode_text(self.model, tokens).float().detach()
                teacher = encode_text(self.model, tokens, teacher=True).float().detach()
            chunks.append({"tokens": tokens, "state": state, "text": text, "start": start, "end": end})
            student_parts.append(text)
            teacher_parts.append(teacher)
            token_slots += tokens["input_ids"].numel()
        if entries:
            student = torch.cat(student_parts)
            teacher = torch.cat(teacher_parts)
            offset = 0
            for entry in entries:
                end = offset + len(entry["captions"])
                entry["text"] = student[offset:end]
                entry["teacher"] = teacher[offset:end]
                offset = end
        return chunks, token_slots, len(captions)

    def replay_organ_text(self, groups, chunks, scaler):
        entries = [entry for group in groups.values() for entry in group]
        if not any("text_grad" in entry for entry in entries):
            return
        # 无有效对比损失的器官以零梯度占位，确保跨 entry 分块的索引仍然对齐。
        gradient = torch.cat([entry["text_grad"] if "text_grad" in entry else torch.zeros_like(entry["text"])
                              for entry in entries])
        for chunk in chunks:
            with chunk["state"].replay(), self.autocast():
                text = encode_text(self.model, chunk["tokens"])
                if self.verify_replay:
                    torch.testing.assert_close(text.float(), chunk["text"], rtol=1e-3, atol=1e-4)
                surrogate = (text.float() * gradient[chunk["start"]:chunk["end"]]).sum()
            scaler.scale(surrogate).backward()

    def backward(self, samples, branch, scaler):
        model = self.model
        started = time.monotonic()
        visual_records = []
        groups = {}
        text_token_slots = 0
        text_sequences = 0
        text_chunks = []
        text_cache_batches = 0
        batched_text = branch == "organ" and self.organ_text_batch_size > 0
        case_count = sum(len(sample["image"]) for sample in samples)
        if branch == "organ":
            model._momentum_update()
        logging.info("gradcache branch=%s phase=cache_visual microbatches=%d cases=%d", branch, len(samples), case_count)
        with torch.no_grad():
            model.temp.clamp_(model.temperature_min, model.temperature_max)
            for sample in samples:
                validate_sample(sample)
                device_sample = {"image": sample["image"].cuda()}
                state = RandomState.capture()
                buffers = snapshot_buffers(model.visual_encoder)
                with self.autocast():
                    encoded, probabilities = encode_visual(model, device_sample, branch)
                entries = {}
                for key, (image, patients) in encoded.items():
                    positions = patients.tolist()
                    captions = [sample["text_input"]["report" if key == "global" else key][position] for position in positions]
                    abnormal = torch.zeros(len(positions), dtype=torch.bool, device="cuda")
                    if key != "global":
                        abnormal = sample["organ_abnormal_flags"][positions, model.organs.index(key)].cuda()
                    entry = {"image": image.float().detach(), "captions": captions, "abnormal": abnormal}
                    groups.setdefault(key, []).append(entry)
                    entries[key] = entry
                visual_records.append((sample, state, buffers, entries))
                del encoded, probabilities, device_sample
            logging.info("gradcache branch=%s phase=cache_text groups=%d", branch, len(groups))
            if batched_text:
                text_chunks, text_token_slots, text_sequences = self.cache_organ_text(groups)
                text_cache_batches = len(text_chunks)
            else:
                for key, entries in groups.items():
                    for entry in entries:
                        tokens = self.text_input(entry["captions"], organ=key != "global")
                        text_token_slots += tokens["input_ids"].numel()
                        text_sequences += len(entry["captions"])
                        text_cache_batches += 1
                        entry["tokens"] = tokens
                        entry["text_state"] = RandomState.capture()
                        with self.autocast():
                            entry["text"] = encode_text(model, tokens).float().detach()
                            if key != "global":
                                entry["teacher"] = encode_text(model, tokens, teacher=True).float().detach()
        final_buffers = snapshot_buffers(model.visual_encoder)
        local_groups = groups
        groups = gather_groups(local_groups)
        metrics = {"branch": branch, "cases": case_count * world_size(), "organs": {}, "effective_organs": 0}
        metrics["text_cache_batches"] = text_cache_batches
        metrics["text_token_slots"] = text_token_slots
        metrics["text_sequences"] = text_sequences
        metrics["text_mean_padded_length"] = text_token_slots / max(text_sequences, 1)
        metrics["cache_seconds"] = time.monotonic() - started
        total_itc = model.temp * 0
        for key, entries in groups.items():
            if "text" not in entries[0]:
                continue
            image = torch.cat([entry["image"] for entry in entries]).requires_grad_()
            text = torch.cat([entry["text"] for entry in entries]).requires_grad_()
            abnormal = torch.cat([entry["abnormal"] for entry in entries])
            captions = sum([entry["captions"] for entry in entries], [])
            if key == "global":
                targets = torch.eye(len(image), device="cuda")
            else:
                if not abnormal.any() or len(image) < 2:
                    continue
                teacher = torch.cat([entry["teacher"] for entry in entries])
                targets = organ_targets(teacher, abnormal, captions, model.temp)
                if torch.unique(targets).numel() == 1:
                    continue
            loss = self.itc_weight * symmetric_contrastive(image, text, model.temp, targets)
            gradients = torch.autograd.grad(loss, (image, text, model.temp))
            total_itc = total_itc + gradients[2].detach() * model.temp / world_size()
            offset = 0
            for entry in entries:
                length = len(entry["image"])
                entry["image_grad"] = gradients[0][offset:offset + length].detach()
                entry["text_grad"] = gradients[1][offset:offset + length].detach()
                offset += length
            stats = alignment_metrics(image.detach(), text.detach(), model.temp.detach())
            stats["loss_itc"] = loss.item()
            if key == "global":
                metrics.update(stats)
            else:
                metrics["organs"][key] = stats
                metrics["effective_organs"] += 1
        if branch == "organ":
            metrics["loss_itc"] = sum(stats["loss_itc"] for stats in metrics["organs"].values())
            metrics["pairs"] = sum(stats["pairs"] for stats in metrics["organs"].values())
            metrics["loss_itc_per_organ"] = metrics["loss_itc"] / max(metrics["effective_organs"], 1)
        logging.info("gradcache branch=%s phase=replay_text effective_organs=%d", branch, metrics["effective_organs"])
        scaler.scale(total_itc).backward()
        del total_itc
        if batched_text:
            self.replay_organ_text(local_groups, text_chunks, scaler)
        else:
            for entries in local_groups.values():
                for entry in entries:
                    if "text_grad" not in entry:
                        continue
                    with entry["text_state"].replay(), self.autocast():
                        text = encode_text(model, entry["tokens"])
                        if self.verify_replay:
                            torch.testing.assert_close(text.float(), entry["text"], rtol=1e-3, atol=1e-4)
                        surrogate = (text.float() * entry["text_grad"]).sum()
                    scaler.scale(surrogate).backward()
        segmentation_loss = 0.0
        logging.info("gradcache branch=%s phase=replay_visual", branch)
        try:
            for sample, state, buffers, entries in visual_records:
                restore_buffers(model.visual_encoder, buffers)
                device_sample = {"image": sample["image"].cuda()}
                with state.replay(), self.autocast():
                    encoded, probabilities = encode_visual(model, device_sample, branch)
                    target = functional.interpolate(sample["seg"].cuda().unsqueeze(1), size=probabilities.shape[-3:], mode="nearest")
                    loss_seg = self.seg_weight * model.dice_loss(probabilities.float(), target)
                    surrogate = loss_seg * (len(sample["image"]) / (case_count * world_size()))
                    segmentation_loss += surrogate.detach().item()
                    for key, (image, _) in encoded.items():
                        entry = entries[key]
                        if self.verify_replay:
                            torch.testing.assert_close(image.float(), entry["image"], rtol=1e-3, atol=1e-4)
                        if "image_grad" in entry:
                            surrogate = surrogate + (image.float() * entry["image_grad"]).sum()
                scaler.scale(surrogate).backward()
                del encoded, probabilities, target, surrogate, device_sample
        finally:
            restore_buffers(model.visual_encoder, final_buffers)
        if world_size() > 1:
            total_segmentation = torch.tensor(segmentation_loss, device="cuda", dtype=torch.float64)
            distributed.all_reduce(total_segmentation)
            segmentation_loss = total_segmentation.item()
        metrics["loss_seg"] = segmentation_loss
        metrics["loss"] = metrics["loss_itc"] + segmentation_loss
        return metrics
