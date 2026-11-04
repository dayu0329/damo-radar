from copy import deepcopy

import torch
from torch import nn
import torch.nn.functional as functional

import gradient_cache as implementation


class TokenBatch(dict):
    def to(self, device):
        return TokenBatch({name: tensor.to(device) for name, tensor in self.items()})


class ToyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(64, 8)
        self.dropout = nn.Dropout(0.2)

    def forward_text(self, tokens):
        hidden = self.dropout(self.embedding(tokens["input_ids"]))
        return type("Output", (), {"last_hidden_state": hidden})


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual_encoder = nn.Sequential(nn.Linear(4, 8), nn.BatchNorm1d(8), nn.Dropout(0.2))
        self.segmentation = nn.Linear(8, 37)
        self.text_encoder = ToyEncoder()
        self.text_proj = nn.Linear(8, 8)
        self.text_encoder_m = deepcopy(self.text_encoder).requires_grad_(False)
        self.text_proj_m = deepcopy(self.text_proj).requires_grad_(False)
        self.temp = nn.Parameter(torch.tensor(0.2))
        self.organs = ["liver", "kidney"]
        self.max_txt_len = 1

    def tokenizer(self, captions, **kwargs):
        return TokenBatch(input_ids=torch.tensor([[int(caption)] for caption in captions]))

    def _momentum_update(self):
        with torch.no_grad():
            for student, teacher in [(self.text_encoder, self.text_encoder_m), (self.text_proj, self.text_proj_m)]:
                for parameter, target in zip(student.parameters(), teacher.parameters()):
                    target.lerp_(parameter, 0.005)

    def dice_loss(self, probabilities, target):
        return -probabilities[:, 0].mean()


def toy_visual(model, sample, branch):
    hidden = model.visual_encoder(sample["image"])
    image = functional.normalize(hidden, dim=-1)
    positions = torch.arange(len(image), device=image.device)
    keys = ["global"] if branch == "global" else model.organs
    groups = {key: (image if key != "kidney" else image.roll(1, dims=-1), positions) for key in keys}
    probabilities = model.segmentation(hidden).softmax(1)[:, :, None, None, None]
    return groups, probabilities


def direct_backward(model, samples, branch):
    if branch == "organ":
        model._momentum_update()
    with torch.no_grad():
        model.temp.clamp_(0.001, 0.5)
    groups = {}
    segmentation_loss = 0
    for sample in samples:
        encoded, probabilities = toy_visual(model, {"image": sample["image"].cuda()}, branch)
        segmentation_loss = segmentation_loss + model.dice_loss(probabilities, None) / len(samples)
        for key, (image, _) in encoded.items():
            captions = sample["text_input"]["report" if key == "global" else key]
            groups.setdefault(key, []).append({"image": image, "captions": captions, "sample": sample})
    for key, entries in groups.items():
        for entry in entries:
            tokens = model.tokenizer(entry["captions"]).to("cuda")
            entry["text"] = implementation.encode_text(model, tokens)
            if key != "global":
                with torch.no_grad():
                    entry["teacher"] = implementation.encode_text(model, tokens, teacher=True)
    total_itc = 0
    for key, entries in groups.items():
        image = torch.cat([entry["image"] for entry in entries])
        text = torch.cat([entry["text"] for entry in entries])
        if key == "global":
            targets = torch.eye(len(image), device="cuda")
        else:
            teacher = torch.cat([entry["teacher"] for entry in entries])
            abnormal = torch.cat([entry["sample"]["organ_abnormal_flags"][:, model.organs.index(key)] for entry in entries]).cuda()
            captions = sum([entry["captions"] for entry in entries], [])
            targets = implementation.organ_targets(teacher, abnormal, captions, model.temp)
        total_itc += implementation.symmetric_contrastive(image, text, model.temp, targets)
    (total_itc + segmentation_loss).backward()
    return total_itc.item(), segmentation_loss.item()


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for cache RNG and gradient comparison")
    implementation.encode_visual = toy_visual
    torch.manual_seed(17)
    samples = []
    for chunk in range(24):
        captions = [str(chunk * 2), str(chunk * 2 + 1)]
        samples.append({
            "image": torch.randn(2, 4), "seg": torch.zeros(2, 1, 1, 1),
            "image_path": [f"case_{caption}" for caption in captions],
            "text_input": {"report": captions, "liver": captions, "kidney": captions},
            "organ_abnormal_flags": torch.tensor([[True, False], [False, True]]),
        })
    for branch in ("global", "organ"):
        reference = ToyModel().cuda().train()
        cached = deepcopy(reference)
        random_state = implementation.RandomState.capture()
        with random_state.replay():
            expected = direct_backward(reference, samples, branch)
        with random_state.replay():
            scaler = torch.amp.GradScaler("cuda", enabled=False)
            measured = implementation.GradientCache(cached, "fp32", verify_replay=True).backward(samples, branch, scaler)
        torch.testing.assert_close(torch.tensor(expected), torch.tensor([measured["loss_itc"], measured["loss_seg"]]), rtol=1e-5, atol=1e-6)
        maximum_error = 0.0
        for (name, left), (_, right) in zip(reference.named_parameters(), cached.named_parameters()):
            if left.grad is None:
                if right.grad is not None:
                    raise AssertionError(f"Unexpected gradient: {name}")
                continue
            torch.testing.assert_close(left.grad, right.grad, rtol=2e-4, atol=2e-5, msg=name)
            maximum_error = max(maximum_error, (left.grad - right.grad).abs().max().item())
        for (name, left), (_, right) in zip(reference.named_buffers(), cached.named_buffers()):
            torch.testing.assert_close(left, right, msg=name)
        print(f"PASS {branch}: 48-case direct/cache losses, gradients, RNG replay and BatchNorm buffers agree; max_gradient_error={maximum_error:.3g}", flush=True)


if __name__ == "__main__":
    main()
