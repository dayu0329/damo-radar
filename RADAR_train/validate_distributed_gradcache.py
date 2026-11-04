import os
from copy import deepcopy

import torch
import torch.distributed as distributed

import gradient_cache as implementation
from validate_gradcache import ToyModel, toy_visual


def sparse_visual(model, sample, branch):
    groups, probabilities = toy_visual(model, sample, branch)
    if branch == "organ" and sample["image"][0, 0].item() == 0:
        groups.pop("kidney")
    return groups, probabilities


def reference_backward(model, samples, branch):
    if branch == "organ":
        model._momentum_update()
    groups = {}
    segmentation_loss = 0
    for sample in samples:
        encoded, probabilities = sparse_visual(model, {"image": sample["image"].cuda()}, branch)
        segmentation_loss = segmentation_loss + model.dice_loss(probabilities, None) / len(samples)
        for key, (image, positions) in encoded.items():
            captions = [sample["text_input"]["report" if key == "global" else key][position] for position in positions.tolist()]
            tokens = model.tokenizer(captions).to("cuda")
            entry = {"image": image, "text": implementation.encode_text(model, tokens), "captions": captions}
            if key != "global":
                entry["abnormal"] = sample["organ_abnormal_flags"][positions.cpu(), model.organs.index(key)].cuda()
                with torch.no_grad():
                    entry["teacher"] = implementation.encode_text(model, tokens, teacher=True)
            groups.setdefault(key, []).append(entry)
    loss = segmentation_loss
    for key, entries in groups.items():
        image = torch.cat([entry["image"] for entry in entries])
        text = torch.cat([entry["text"] for entry in entries])
        if key == "global":
            targets = torch.eye(len(image), device="cuda")
        else:
            abnormal = torch.cat([entry["abnormal"] for entry in entries])
            if not abnormal.any() or len(image) < 2:
                continue
            teacher = torch.cat([entry["teacher"] for entry in entries])
            targets = implementation.organ_targets(teacher, abnormal, sum([entry["captions"] for entry in entries], []), model.temp)
            if torch.unique(targets).numel() == 1:
                continue
        loss = loss + implementation.symmetric_contrastive(image, text, model.temp, targets)
    loss.backward()
    return loss.item()


def main():
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")) % torch.cuda.device_count())
    distributed.init_process_group("nccl", device_id=torch.device("cuda", torch.cuda.current_device()))
    rank = distributed.get_rank()
    count = distributed.get_world_size()
    if 48 % (2 * count):
        raise ValueError("Test requires 48 divisible by 2 * world size")
    implementation.encode_visual = sparse_visual
    torch.manual_seed(17)
    samples = []
    for owner in range(count):
        for chunk in range(24 // count):
            first = owner * (48 // count) + chunk * 2
            captions = [str(first), str(first + 1)]
            image = torch.randn(2, 4)
            image[:, 0] = owner
            samples.append({
                "image": image, "seg": torch.zeros(2, 1, 1, 1), "image_path": captions,
                "text_input": {"report": captions, "liver": captions, "kidney": captions},
                "organ_abnormal_flags": torch.tensor([[True, False], [False, True]]),
            })
    for branch in ("global", "organ"):
        torch.manual_seed(29)
        reference = ToyModel().cuda().eval()
        cached = deepcopy(reference)
        expected = reference_backward(reference, samples, branch)
        local = samples[rank * (24 // count):(rank + 1) * (24 // count)]
        measured = implementation.GradientCache(cached, "fp32", True).backward(
            local, branch, torch.amp.GradScaler("cuda", enabled=False),
        )
        implementation.synchronize_gradients(cached)
        torch.testing.assert_close(torch.tensor(measured["loss"]), torch.tensor(expected), rtol=1e-5, atol=1e-6)
        for (name, expected_parameter), (_, parameter) in zip(reference.named_parameters(), cached.named_parameters()):
            if expected_parameter.grad is None:
                assert parameter.grad is None, name
            else:
                torch.testing.assert_close(parameter.grad, expected_parameter.grad, rtol=3e-4, atol=3e-5, msg=name)
        for model in (reference, cached):
            torch.optim.AdamW(model.parameters(), lr=1e-4).step()
        for (name, expected_parameter), (_, parameter) in zip(reference.named_parameters(), cached.named_parameters()):
            torch.testing.assert_close(parameter, expected_parameter, rtol=1e-5, atol=1e-6, msg=name)
        if rank == 0:
            print(f"PASS {count} ranks {branch}: 48-case loss, gradients and optimizer update match direct reference", flush=True)
    distributed.destroy_process_group()


if __name__ == "__main__":
    main()
