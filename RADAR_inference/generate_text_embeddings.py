import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as functional
from transformers import BertConfig, BertTokenizer


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "RADAR_train"))


def default_prompts():
    source = Path(__file__).with_name("inference_demo.py")
    tree = ast.parse(source.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Attribute) and target.attr == "english_mapping"
            for target in node.targets
        ):
            mapping = ast.literal_eval(node.value)
            break
    else:
        raise ValueError("Cannot find english_mapping in inference_demo.py")
    result = {}
    for key, label in mapping.items():
        organ, disease = label.split("_", 1)
        result[key] = {
            "negative_prompts": [f"{organ.lower()}: no {disease.lower()}.", "normal."],
            "positive_prompts": [f"{organ.lower()}: {disease.lower()}."],
        }
    return result


def merlin_prompts(source):
    variables = {}
    result = {}
    tree = ast.parse(source.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and isinstance(node.value, ast.Call):
            call = node.value
            if isinstance(call.func, ast.Name) and call.func.id == "DiseasePrompts":
                variables[target.id] = {keyword.arg: ast.literal_eval(keyword.value) for keyword in call.keywords}
        elif (
            isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name)
            and target.value.id == "disease_prompts" and isinstance(node.value, ast.Name)
        ):
            disease = ast.literal_eval(target.slice)
            entry = variables[node.value.id]
            key = entry["region"] + "_" + disease.replace("_", "-")
            result[key] = {
                "positive_prompts": [prompt.replace("\\", "") for prompt in entry["positive_prompts"]],
                "negative_prompts": [prompt.replace("\\", "") for prompt in entry["negative_prompts"]] + ["normal"],
            }
    if not result:
        raise ValueError(f"No MERLIN DiseasePrompts found in {source}")
    return result


def validate_prompts(prompts):
    if not isinstance(prompts, dict) or not prompts:
        raise ValueError("Prompts must be a nonempty JSON object keyed by inference item")
    for key, entry in prompts.items():
        if not isinstance(key, str) or not isinstance(entry, dict):
            raise ValueError(f"Invalid prompt entry: {key!r}")
        for polarity in ("negative_prompts", "positive_prompts"):
            values = entry.get(polarity)
            if not isinstance(values, list) or not values or any(
                not isinstance(value, str) or not value.strip() for value in values
            ):
                raise ValueError(f"{key}: {polarity} must contain nonempty strings")


def load_text_model(checkpoint_path, bert_dir, config_path, device):
    from lavis.models.med import XBertEncoder

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
    state = checkpoint.get("model", checkpoint)
    state = {key.removeprefix("module."): value for key, value in state.items()}
    encoder_state = {key[len("text_encoder."):]: value for key, value in state.items() if key.startswith("text_encoder.")}
    projection_state = {key[len("text_proj."):]: value for key, value in state.items() if key.startswith("text_proj.")}
    if not encoder_state or set(projection_state) != {"weight", "bias"}:
        raise ValueError("Checkpoint must contain student text_encoder and text_proj weights")
    tokenizer = BertTokenizer.from_pretrained(str(bert_dir), local_files_only=True)
    config = BertConfig.from_json_file(str(config_path))
    embedding_key = "embeddings.word_embeddings.weight"
    if embedding_key not in encoder_state:
        raise ValueError("Checkpoint is not a compatible RADAR XBertEncoder")
    if len(tokenizer) != encoder_state[embedding_key].shape[0]:
        raise ValueError("Tokenizer vocabulary size differs from checkpoint; use the training tokenizer")
    config.vocab_size = len(tokenizer)
    encoder = XBertEncoder(config=config, add_pooling_layer=False)
    message = encoder.load_state_dict(encoder_state, strict=False)
    allowed_extra = {"cls.predictions.decoder.weight", "cls.predictions.decoder.bias"}
    if message.missing_keys or set(message.unexpected_keys) - allowed_extra:
        raise ValueError(f"Incomplete/incompatible text encoder: {message}")
    projection = torch.nn.Linear(projection_state["weight"].shape[1], projection_state["weight"].shape[0])
    projection.load_state_dict(projection_state, strict=True)
    encoder.to(device).eval()
    projection.to(device).eval()
    return encoder, projection, tokenizer, checkpoint.get("epoch")


@torch.inference_mode()
def generate(prompts, encoder, projection, tokenizer, device, batch_size, max_length, aggregation="normalized-mean"):
    result = {}
    for key, entry in prompts.items():
        features = []
        for polarity in ("negative_prompts", "positive_prompts"):
            captions = entry[polarity]
            chunks = []
            for offset in range(0, len(captions), batch_size):
                tokens = tokenizer(
                    captions[offset:offset + batch_size], padding="max_length", truncation=True,
                    max_length=max_length, return_tensors="pt",
                ).to(device)
                hidden = encoder.forward_text(tokens).last_hidden_state[:, 0]
                chunks.append(functional.normalize(projection(hidden).float(), dim=-1))
            mean = torch.cat(chunks).mean(0)
            if not torch.isfinite(mean).all() or mean.norm() < 1e-8:
                raise ValueError(f"Invalid text embedding for {key}: {polarity}")
            features.append((functional.normalize(mean, dim=0) if aggregation == "normalized-mean" else mean).cpu())
        result[key] = torch.stack(features)
    return result


def main():
    parser = argparse.ArgumentParser(description="Export checkpoint-matched RADAR negative/positive text prototypes")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--prompts", type=Path, help="JSON with negative_prompts and positive_prompts per inference key")
    parser.add_argument("--create-prompts", type=Path, help="Write editable English templates and exit; these are not recovered official prompts")
    parser.add_argument("--prompt-source", choices=("templates", "merlin"), default="templates")
    parser.add_argument("--merlin-source", type=Path, default=ROOT / "RADAR_train/infer_merlin_anatomy.py")
    parser.add_argument("--aggregation", choices=("mean", "normalized-mean"), help="Default: mean for MERLIN (matches original logit averaging), normalized-mean for templates")
    parser.add_argument("--bert-dir", type=Path, default=ROOT / "ckpt/bert-base-uncased")
    parser.add_argument("--config", type=Path, help="Training BERT config; defaults to BERT_DIR/config.json")
    parser.add_argument("--reference", type=Path, help="Optional original .pt: verify all keys and tensor shapes")
    parser.add_argument("--device", default="cpu", help="cpu or cuda:N; output tensors are always stored on CPU")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, help="Default: 100 for MERLIN, 512 for templates")
    args = parser.parse_args()
    args.max_length = args.max_length if args.max_length is not None else (100 if args.prompt_source == "merlin" else 512)
    aggregation = args.aggregation or ("mean" if args.prompt_source == "merlin" else "normalized-mean")
    source_prompts = lambda: merlin_prompts(args.merlin_source) if args.prompt_source == "merlin" else default_prompts()
    if args.create_prompts:
        args.create_prompts.parent.mkdir(parents=True, exist_ok=True)
        with args.create_prompts.open("x", encoding="utf-8") as handle:
            json.dump(source_prompts(), handle, ensure_ascii=False, indent=2)
        print(f"Wrote {args.prompt_source} prompts to {args.create_prompts}")
        if args.prompt_source == "templates":
            print("These generated templates are NOT the official prompt ensemble.")
        return
    if not args.checkpoint or not args.output or (not args.prompts and args.prompt_source != "merlin"):
        parser.error("Export requires --checkpoint, --output and either --prompts or --prompt-source merlin")
    if args.batch_size < 1 or args.max_length < 1:
        parser.error("batch-size and max-length must be positive")
    metadata_path = args.output.with_suffix(args.output.suffix + ".json")
    if args.output.exists() or metadata_path.exists():
        raise FileExistsError("Output already exists; choose a new output name")
    prompts = json.loads(args.prompts.read_text(encoding="utf-8-sig")) if args.prompts else source_prompts()
    validate_prompts(prompts)
    output_keys = {key: tuple(key.split("_", 1)) if args.prompt_source == "merlin" else key for key in prompts}
    if args.prompt_source == "merlin" and any(len(key) != 2 for key in output_keys.values()):
        raise ValueError("MERLIN keys must use region_disease format")
    config_path = args.config or args.bert_dir / "config.json"
    encoder, projection, tokenizer, epoch = load_text_model(
        args.checkpoint, args.bert_dir, config_path, torch.device(args.device),
    )
    if args.max_length > encoder.config.max_position_embeddings:
        raise ValueError("max-length exceeds the encoder position embedding limit")
    reference = None
    if args.reference:
        reference = torch.load(args.reference, map_location="cpu", weights_only=False)
        if set(reference) != set(output_keys.values()):
            raise ValueError(f"Prompt/reference key mismatch: missing={set(reference) - set(output_keys.values())}, extra={set(output_keys.values()) - set(reference)}")
    generated = generate(prompts, encoder, projection, tokenizer, torch.device(args.device), args.batch_size, args.max_length, aggregation)
    features = {output_keys[key]: value for key, value in generated.items()}
    if reference is not None:
        for key, feature in features.items():
            if feature.shape != reference[key].shape:
                raise ValueError(f"Shape mismatch for {key}: {feature.shape} != {reference[key].shape}")
    metadata = {
        "checkpoint": str(args.checkpoint.resolve()), "checkpoint_epoch_zero_based": epoch,
        "tokenizer": str(args.bert_dir.resolve()), "config": json.loads(config_path.read_text()),
        "vocab_sha256": hashlib.sha256((args.bert_dir / "vocab.txt").read_bytes()).hexdigest(),
        "class_order": ["negative", "positive"], "max_length": args.max_length,
        "pooling": "CLS -> student projection -> L2 -> mean per class" + (" -> L2" if aggregation == "normalized-mean" else ""),
        "prompt_source": args.prompt_source, "aggregation": aggregation,
        "merlin_source": str(args.merlin_source.resolve()) if args.prompt_source == "merlin" else None,
        "prompts": prompts, "torch": torch.__version__,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    torch.save(features, temporary)
    temporary.replace(args.output)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(features)} items, each [negative, positive] x {projection.out_features}, to {args.output}")


if __name__ == "__main__":
    main()
