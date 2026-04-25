#!/usr/bin/env python3
"""
Script to merge a LoRA adapter into the base model.
This creates a standalone model that can be used for further fine-tuning or inference.
"""

import os
import json
import torch
import gc
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL_PATH = "./outputs/models/base_model"
ADAPTER_PATH = "./outputs/models/final_adapter"
MERGED_OUTPUT_PATH = "./outputs/models/final_model_merged"


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (project_root() / path).resolve()


def is_adapter_dir(path: Path) -> bool:
    return (path / "adapter_config.json").exists() and (path / "adapter_model.safetensors").exists()


def find_latest_adapter_checkpoint(search_root: Path) -> Path:
    if not search_root.exists():
        return None

    checkpoints = []
    for child in search_root.iterdir():
        if not child.is_dir() or not child.name.startswith("checkpoint-"):
            continue
        if not is_adapter_dir(child):
            continue
        try:
            step = int(child.name.split("-")[1])
        except (IndexError, ValueError):
            continue
        checkpoints.append((step, child))

    if not checkpoints:
        return None

    checkpoints.sort(key=lambda item: item[0])
    return checkpoints[-1][1]

def main():
    base_model_path = resolve_path(BASE_MODEL_PATH)
    adapter_path = resolve_path(ADAPTER_PATH)
    merged_output_path = resolve_path(MERGED_OUTPUT_PATH)

    print("=" * 60)
    print("MERGING LORA ADAPTER")
    print(f"Base Model: {base_model_path}")
    print(f"Adapter:    {adapter_path}")
    print(f"Output:     {merged_output_path}")
    print("=" * 60)

    if not is_adapter_dir(adapter_path):
        latest = find_latest_adapter_checkpoint(resolve_path("./outputs/models"))
        if latest is not None:
            print(f"⚠️ Adapter at configured path is missing or incomplete. Falling back to latest checkpoint: {latest}")
            adapter_path = latest
        else:
            print(f"❌ Error: No valid adapter directory found at {adapter_path}")
            print("Expected files: adapter_config.json and adapter_model.safetensors")
            return

    adapter_cfg_path = adapter_path / "adapter_config.json"
    try:
        with open(adapter_cfg_path, "r", encoding="utf-8") as f:
            adapter_cfg = json.load(f)
    except Exception as exc:
        print(f"❌ Error: Cannot read adapter config: {adapter_cfg_path}")
        print(f"Details: {exc}")
        return

    config_base = adapter_cfg.get("base_model_name_or_path")
    if not base_model_path.exists() and config_base:
        cfg_base_path = resolve_path(config_base)
        if cfg_base_path.exists():
            print(f"⚠️ Configured base model path not found. Using adapter config base model path: {cfg_base_path}")
            base_model_path = cfg_base_path

    if not base_model_path.exists():
        print(f"❌ Error: Base model path does not exist: {base_model_path}")
        return

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer_source = adapter_path if (adapter_path / "tokenizer_config.json").exists() else base_model_path
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_source),
        trust_remote_code=True,
        local_files_only=True
    )

    # Load base model in 16-bit to avoid losing precision during the merge.
    # Note: Do not load in 8-bit or 4-bit if you intend to merge. We need the actual weights.
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        str(base_model_path),
        device_map="cpu", # Load to RAM first, or "auto" if you have enough VRAM
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=True,
        low_cpu_mem_usage=True
    )

    # Load the LoRA adapter
    print("Loading PEFT adapter...")
    model = PeftModel.from_pretrained(
        base_model,
        str(adapter_path),
        device_map="cpu",
        torch_dtype=torch.bfloat16
    )

    # Merge weights and unload adapter
    print("Merging adapter into base model (this might take a minute)...")
    model = model.merge_and_unload()

    # Save the merged model and tokenizer
    print(f"Saving merged model to {merged_output_path}...")
    os.makedirs(merged_output_path, exist_ok=True)
    model.save_pretrained(str(merged_output_path), safe_serialization=True)
    tokenizer.save_pretrained(str(merged_output_path))

    print("\nMerge complete!")
    
    # Cleanup memory
    del model
    del base_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
