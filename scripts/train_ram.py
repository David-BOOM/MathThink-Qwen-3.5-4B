#!/usr/bin/env python3
"""
QLoRA Fine-Tuning Script for Qwen 3.5 4B
OPTIMIZED VERSION - Target 18-20GB VRAM usage

Features:
- Batch size 4 with gradient accumulation 2 (effective batch 8)
- Max sequence length 768
- Target ~18-20GB VRAM for RTX 4090
- Frequent checkpoints (every 500 steps)
- Memory monitoring and crash recovery
"""

import os
import sys
import json
import torch
import gc
from datetime import datetime
from pathlib import Path
from typing import Optional

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    TrainerCallback,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
from datasets import load_dataset


# Configuration - OPTIMIZED for 18-20GB VRAM
MODEL_PATH = "./outputs/models/base_model"
DATA_PATH = "./Nemotron-Math-v3/train_processed.jsonl"
OUTPUT_DIR = "./outputs/models"
MAX_SEQ_LENGTH = 768  # Increased from 512 for better learning


class MemoryMonitorCallback(TrainerCallback):
    """Monitor memory and warn if getting too high."""
    
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 100 == 0:
            if torch.cuda.is_available():
                used = torch.cuda.memory_allocated() / 1024**3
                total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                pct = used / total * 100
                print(f"  [Step {state.global_step}] GPU Memory: {used:.1f}GB / {total:.1f}GB ({pct:.0f}%)")
                
                # Force garbage collection periodically
                if state.global_step % 500 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()


def get_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Find latest checkpoint for crash recovery."""
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        return None
    
    checkpoints = [d for d in checkpoint_path.iterdir() 
                   if d.is_dir() and d.name.startswith("checkpoint-")]
    
    if not checkpoints:
        return None
    
    checkpoints.sort(key=lambda x: int(x.name.split("-")[1]))
    latest = checkpoints[-1]
    print(f"✓ Resuming from: {latest.name}")
    return str(latest)


def save_training_state(output_dir: str, step: int, total_steps: int, error: str = None):
    """Save training state."""
    state = {
        "timestamp": datetime.now().isoformat(),
        "last_step": step,
        "total_steps": total_steps,
        "complete": step >= total_steps,
        "error": error
    }
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(output_dir) / "training_state.json", "w") as f:
        json.dump(state, f, indent=2)


def load_model_and_tokenizer(model_path: str):
    """Load model with conservative memory settings."""
    print(f"Loading model from: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side="right",
        local_files_only=True,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 4-bit quantization for low memory
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        local_files_only=True,
        low_cpu_mem_usage=True,  # Reduce CPU memory
    )
    
    model = prepare_model_for_kbit_training(model)
    
    lora_config = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    model = get_peft_model(model, lora_config)
    model.config.use_cache = False
    
    # Enable gradient checkpointing early
    model.gradient_checkpointing_enable()
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"✓ Trainable: {trainable:,} ({100*trainable/total:.2f}%)")
    
    return model, tokenizer


def load_streaming_dataset(data_path: str, tokenizer):
    """Load dataset with streaming/lazy loading to reduce RAM usage."""
    print(f"Loading dataset (streaming mode): {data_path}")
    
    # Load with HuggingFace datasets - it handles memory better
    dataset = load_dataset("json", data_files=data_path, split="train")
    print(f"✓ Dataset: {len(dataset):,} examples")
    
    def format_example(example):
        messages = example.get("messages", [])
        try:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
        except:
            text = ""
        return {"text": text}
    
    # Process in batches to avoid memory spike
    dataset = dataset.map(
        format_example,
        remove_columns=dataset.column_names,
        num_proc=1,  # Single process to avoid memory multiplication
        load_from_cache_file=True,  # Use disk cache
    )
    
    # Filter empty
    dataset = dataset.filter(lambda x: len(x["text"]) > 0)
    print(f"✓ After filtering: {len(dataset):,} examples")
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding="max_length",
        )
    
    # Tokenize with disk caching
    dataset = dataset.map(
        tokenize_function,
        batched=True,
        batch_size=1000,
        remove_columns=["text"],
        num_proc=1,
        load_from_cache_file=True,
    )
    
    return dataset


def train():
    """Main training - OPTIMIZED for 18-20GB VRAM."""
    print("=" * 60)
    print("QLoRA FINE-TUNING: Qwen 3.5 4B")
    print("OPTIMIZED MODE - Target 18-20GB VRAM")
    print("=" * 60)
    
    # Clear GPU memory before starting
    gc.collect()
    torch.cuda.empty_cache()
    
    # Check for resume
    resume_checkpoint = get_latest_checkpoint(OUTPUT_DIR)
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(MODEL_PATH)
    
    # Check memory after model load
    if torch.cuda.is_available():
        used = torch.cuda.memory_allocated() / 1024**3
        print(f"GPU memory after model load: {used:.1f}GB")
    
    # Load dataset with streaming
    dataset = load_streaming_dataset(DATA_PATH, tokenizer)
    
    # Optimized batch settings for 18-20GB VRAM
    batch_size = 4  # Increased from 2
    grad_accum = 2  # Reduced to maintain effective batch of 8
    effective_batch = batch_size * grad_accum
    total_steps = len(dataset) // effective_batch
    
    print(f"\nConfiguration (OPTIMIZED 18-20GB VRAM):")
    print(f"  Dataset: {len(dataset):,} examples")
    print(f"  Batch: {batch_size} x {grad_accum} = {effective_batch}")
    print(f"  Max seq length: {MAX_SEQ_LENGTH}")
    print(f"  Steps: {total_steps:,}")
    print(f"  Checkpoints: every 500 steps")
    print(f"  Resume: {resume_checkpoint or 'scratch'}")
    print("-" * 60)
    
    # Training arguments - OPTIMIZED
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        
        # Optimized batch size for 18-20GB VRAM
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        
        num_train_epochs=1,
        
        bf16=True,
        fp16=False,
        
        optim="adamw_torch",
        weight_decay=0.01,
        max_grad_norm=1.0,
        
        # MORE FREQUENT checkpoints for crash recovery
        save_strategy="steps",
        save_steps=500,
        save_total_limit=5,
        
        logging_steps=50,
        report_to="tensorboard",
        logging_dir=f"{OUTPUT_DIR}/logs",
        
        eval_strategy="no",
        
        # Memory optimization
        gradient_checkpointing=True,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        
        resume_from_checkpoint=resume_checkpoint,
        seed=42,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Create trainer with memory monitor
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        callbacks=[MemoryMonitorCallback()],
    )
    
    save_training_state(OUTPUT_DIR, 0, total_steps)
    
    try:
        print("\n🚀 Starting training (optimized for 18-20GB VRAM)...")
        print("   Memory will be monitored every 100 steps")
        print("   Checkpoints saved every 500 steps")
        trainer.train(resume_from_checkpoint=resume_checkpoint)
        
        print("\n💾 Saving final adapter...")
        trainer.save_model(f"{OUTPUT_DIR}/final_adapter")
        tokenizer.save_pretrained(f"{OUTPUT_DIR}/final_adapter")
        
        save_training_state(OUTPUT_DIR, total_steps, total_steps)
        print("\n" + "=" * 60)
        print("✅ TRAINING COMPLETE!")
        print("=" * 60)
        
    except KeyboardInterrupt:
        step = trainer.state.global_step if hasattr(trainer, 'state') else 0
        print(f"\n⚠️ Interrupted at step {step}. Saving checkpoint...")
        checkpoint_path = f"{OUTPUT_DIR}/checkpoint-{step}"
        trainer.save_model(checkpoint_path)
        trainer.state.save_to_json(f"{checkpoint_path}/trainer_state.json")
        tokenizer.save_pretrained(checkpoint_path)
        print(f"✓ Checkpoint saved to: {OUTPUT_DIR}/checkpoint-{step}")
        save_training_state(OUTPUT_DIR, step, total_steps, "interrupted")
        
    except Exception as e:
        step = trainer.state.global_step if hasattr(trainer, 'state') else 0
        print(f"\n❌ Error at step {step}: {e}")
        save_training_state(OUTPUT_DIR, step, total_steps, str(e))
        raise


if __name__ == "__main__":
    train()
