"""
MATH Benchmark Evaluation Script
Evaluates model on competition mathematics problems.
Metric: Pass@1 accuracy, zero-shot.
"""

import os
import sys
import json
import re
import argparse
import time
import torch
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict
from transformers import StoppingCriteria, StoppingCriteriaList

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from datasets import load_dataset
except ImportError:
    print("Error: datasets not installed. Run: pip install datasets")
    sys.exit(1)


# =============================================================================
# Configuration
# =============================================================================

BENCHMARK_NAME = "MATH"
MATH_DATASETS = [
    ("HuggingFaceH4/MATH-500", None),
]
SPLIT = "test"

# Fallback order when a dataset does not expose the requested split.
FALLBACK_SPLITS = ["test", "validation", "valid", "dev", "train"]

OUTPUT_DIR = "./outputs/results"
SYSTEM_PROMPT = (
    "You are a MATH benchmark solver. "
    "Solve carefully, verify algebra and arithmetic, and provide one final simplified exact answer. "
    "End with: Final answer: \\boxed{...}."
)





import urllib.request
import urllib.error
import urllib.parse
import json

def count_output_tokens(a, b): return 0
def resolve_context_window(a, b): return 4096

def generate_responses_batch(
    model, # Unused
    tokenizer, # Unused
    prompts: List[str],
    context_window: Optional[int] = None,
    max_output_tokens: Optional[int] = None,
    temperature: float = 0.0,
    realtime_progress: bool = False,
    label: str = "Batch",
) -> Tuple[List[str], List[dict]]:
    """Generate responses for a batch of prompts using LM Studio API in parallel."""
    import concurrent.futures
    
    if not prompts: return [], []

    # LM studio is usually running on http://127.0.0.1:65535
    api_base = "http://127.0.0.1:65535/v1/chat/completions"
    
    def fetch_prompt(prompt: str) -> dict:
        payload = {
            "model": "local-model",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 100000,
            "temperature": 1,
            "top_p": 0.95,
            "repetition_penalty": 1.1,
            "stream": False
        }

        req = urllib.request.Request(
            api_base,
            data=json.dumps(payload).encode('utf-8'),
            headers={'Content-Type': 'application/json'}
        )

        try:
            with urllib.request.urlopen(req) as response:
                result = json.loads(response.read().decode('utf-8'))
                
                content = result['choices'][0]['message']['content']
                usage = result.get('usage', {})
                in_tok = usage.get('prompt_tokens', 0)
                out_tok = usage.get('completion_tokens', 0)
                tot_tok = usage.get('total_tokens', 0)
                
                return {
                    "response": content.strip(),
                    "stats": {
                        "input_tokens": in_tok,
                        "output_tokens": out_tok,
                        "total_tokens": tot_tok,
                    }
                }
        except Exception as e:
            print(f"[{label}] API Error: {e}")
            return {
                "response": "",
                "stats": {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                }
            }

    responses = []
    token_stats = []
    
    # Use ThreadPoolExecutor to send requests concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(prompts)) as executor:
        results = list(executor.map(fetch_prompt, prompts))
        
    for res in results:
        responses.append(res["response"])
        token_stats.append(res["stats"])
            
    return responses, token_stats



# =============================================================================
# Answer Extraction & Comparison
# =============================================================================

def extract_math_answer(text: str) -> Optional[str]:
    """
    Extract answer from MATH format (\\boxed{...}).

    Args:
        text: The generated text or ground truth

    Returns:
        Extracted answer
    """
    pattern = r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}"
    matches = re.findall(pattern, text)
    if matches:
        return matches[-1].strip()

    simple_pattern = r"\\boxed\{([^}]+)\}"
    simple_matches = re.findall(simple_pattern, text)
    if simple_matches:
        return simple_matches[-1].strip()

    return None


def normalize_math_answer(answer: str) -> str:
    """
    Normalize MATH answer for comparison.

    Args:
        answer: Raw answer string

    Returns:
        Normalized answer
    """
    if answer is None:
        return ""

    answer = answer.strip()
    answer = re.sub(r"\\text\{([^}]*)\}", r"\1", answer)
    answer = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", answer)
    answer = re.sub(r"\\mathbf\{([^}]*)\}", r"\1", answer)
    answer = re.sub(r"\\left|\\right", "", answer)
    answer = re.sub(r"\\,|\\;|\\:|\\!", "", answer)
    answer = answer.replace("\\dfrac", "\\frac")
    answer = answer.replace("\\tfrac", "\\frac")
    answer = answer.replace(" ", "")
    answer = answer.replace("\\pi", "pi")
    answer = answer.replace("\\infty", "inf")

    return answer.lower()


def answers_equivalent(pred: str, gold: str) -> bool:
    """
    Check if predicted answer is equivalent to gold answer.
    Uses normalization for comparison.

    Args:
        pred: Predicted answer
        gold: Gold answer

    Returns:
        True if answers are equivalent
    """
    if pred is None or gold is None:
        return False

    norm_pred = normalize_math_answer(pred)
    norm_gold = normalize_math_answer(gold)

    if norm_pred == norm_gold:
        return True

    try:
        pred_num = float(eval(norm_pred.replace("^", "**")))
        gold_num = float(eval(norm_gold.replace("^", "**")))
        return abs(pred_num - gold_num) < 1e-6
    except Exception:
        return False


# =============================================================================
# Dataset Loading
# =============================================================================

def load_math_dataset() -> List[dict]:
    """Load MATH test dataset."""
    print(f"Loading {BENCHMARK_NAME} dataset...")

    def _load_with_split_fallback(dataset_name: str, config: Optional[str] = None):
        candidate_splits = [SPLIT] + [s for s in FALLBACK_SPLITS if s != SPLIT]
        last_error = None
        for split_name in candidate_splits:
            try:
                dataset = load_dataset(dataset_name, config, split=split_name)
                if split_name != SPLIT:
                    cfg_text = f", config={config}" if config else ""
                    print(
                        f"  Requested split '{SPLIT}' unavailable for {dataset_name}{cfg_text}; "
                        f"using '{split_name}' instead."
                    )
                return dataset
            except Exception as exc:
                last_error = exc
                message = str(exc)
                # Keep trying only when this looks like a split mismatch.
                if "Unknown split" not in message and "split" not in message.lower():
                    raise
        raise last_error

    problems = []
    running_index = 1

    for dataset_name, configs in MATH_DATASETS:
        try:
            print(f"Trying: {dataset_name}")
            if configs:
                for config in configs:
                    dataset = _load_with_split_fallback(dataset_name, config)
                    for item in dataset:
                        problems.append(
                            {
                                "problem": item.get("problem", ""),
                                "solution": item.get("solution", ""),
                                "level": item.get("level", "unknown"),
                                "type": item.get("type", config),
                                "gold_answer": extract_math_answer(item.get("solution", "")),
                                "number": running_index,
                            }
                        )
                        running_index += 1
            else:
                dataset = _load_with_split_fallback(dataset_name)
                for item in dataset:
                    problems.append(
                        {
                            "problem": item.get("problem", ""),
                            "solution": item.get("solution", ""),
                            "level": item.get("level", "unknown"),
                            "type": item.get("type", "unknown"),
                            "gold_answer": extract_math_answer(item.get("solution", "")),
                            "number": running_index,
                        }
                    )
                    running_index += 1

            if problems:
                print(f" Loaded {len(problems)} problems from {dataset_name}")
                break
        except Exception as exc:
            print(f"  Could not load {dataset_name}: {exc}")
            continue

    if not problems:
        raise RuntimeError("Could not load MATH dataset from configured sources")

    print(f"Loaded {len(problems)} problems")

    level_counts = defaultdict(int)
    type_counts = defaultdict(int)
    for problem in problems:
        level_counts[problem["level"]] += 1
        type_counts[problem["type"]] += 1

    print("\nLevel distribution:")
    for level, count in sorted(level_counts.items()):
        print(f"  {level}: {count}")

    print("\nSubject distribution:")
    for subject, count in sorted(type_counts.items()):
        print(f"  {subject}: {count}")

    return problems


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_math(
    model_path: str,
    lora_path: Optional[str] = None,
    max_samples: Optional[int] = None,
    max_output_tokens: Optional[int] = None,
    temperature: float = 0.0,
    output_file: Optional[str] = None,
    use_baseline: bool = False,
    batch_size: int = 4,
    load_in_4bit: bool = False,
    realtime_token_progress: bool = False,
    token_log_file: Optional[str] = None,
):
    """
    Evaluate model on MATH benchmark.

    Args:
        model_path: Path to the model
        lora_path: Path to LoRA adapters
        max_samples: Maximum samples to evaluate
        max_output_tokens: Maximum generated tokens per question (None = auto)
        temperature: Generation temperature (0.0 = greedy deterministic)
        output_file: Where to save results
        use_baseline: If True, use base model without fine-tuning
        realtime_token_progress: Print per-second token updates per question
        token_log_file: JSONL file for per-question token logs

    Returns:
        Dictionary with evaluation results
    """
    # from scripts.inference import load_model

    print("=" * 60)
    print(f"EVALUATING: {BENCHMARK_NAME}")
    print("=" * 60)
    print(f"Model: {model_path}")
    if lora_path:
        print(f"LoRA: {lora_path}")
    print(f"Baseline mode: {use_baseline}")
    print()

    if use_baseline:
        # For lmstudio evaluation, we don't actually load the model locally
        model, tokenizer = None, None
    else:
        model, tokenizer = None, None

    problems = load_math_dataset()

    if max_samples:
        problems = problems[:max_samples]
        print(f"\nEvaluating on {max_samples} samples")

    if output_file is None:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        suffix = "_baseline" if use_baseline else "_finetuned"
        output_file = os.path.join(OUTPUT_DIR, f"math{suffix}_results.json")

    if token_log_file is None:
        base, ext = os.path.splitext(output_file)
        token_log_file = f"{base}_token_log.jsonl" if ext else f"{output_file}_token_log.jsonl"

    Path(token_log_file).parent.mkdir(parents=True, exist_ok=True)

    context_window = resolve_context_window(model, tokenizer)
    batch_size = max(1, batch_size)
    if max_output_tokens is not None and max_output_tokens < 1:
        raise ValueError("max_output_tokens must be >= 1 when provided")
    if temperature < 0:
        raise ValueError("temperature must be >= 0")

    print(f"Batch size: {batch_size}")
    print(f"Context window: {context_window} tokens (auto max mode)")
    if max_output_tokens is None:
        print("Max output tokens per question: auto (up to remaining context window)")
    else:
        print(f"Max output tokens per question: {max_output_tokens}")
    if temperature == 0:
        print("Temperature: 0.0 (greedy deterministic)")
    else:
        print(f"Temperature: {temperature} (sampling enabled)")
    print(f"4-bit quantization: {load_in_4bit}")
    print(f"Token log file: {token_log_file}")

    results = []
    correct = 0
    total = 0
    total_input_tokens = 0
    total_output_tokens = 0

    level_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    type_stats = defaultdict(lambda: {"correct": 0, "total": 0})

    print("\nRunning evaluation...")
    pbar = tqdm(total=len(problems), desc=BENCHMARK_NAME)
    idx = 0

    with open(token_log_file, "w", encoding="utf-8") as token_log_fp:
        while idx < len(problems):
            current_batch_size = min(batch_size, len(problems) - idx)
            batch = problems[idx : idx + current_batch_size]
            questions = [item["problem"] for item in batch]

            try:
                label = f"Batch Q{idx+1}-{idx+current_batch_size}" if current_batch_size > 1 else f"Q{idx+1}"
                responses, batch_token_stats = generate_responses_batch(
                    model,
                    tokenizer,
                    questions,
                    context_window=context_window,
                    max_output_tokens=max_output_tokens,
                    temperature=temperature,
                    realtime_progress=realtime_token_progress,
                    label=label,
                )
            except torch.cuda.OutOfMemoryError:
                if current_batch_size == 1:
                    raise
                torch.cuda.empty_cache()
                batch_size = max(1, current_batch_size // 2)
                print(f"\nOOM at batch size {current_batch_size}. Reducing to {batch_size} and retrying.")
                continue

            for problem, response, token_stat in zip(batch, responses, batch_token_stats):
                question = problem["problem"]
                gold_answer = problem["gold_answer"]
                level = problem["level"]
                subject = problem["type"]

                pred_answer = extract_math_answer(response)
                is_correct = answers_equivalent(pred_answer, gold_answer)

                if is_correct:
                    correct += 1
                    level_stats[level]["correct"] += 1
                    type_stats[subject]["correct"] += 1

                total += 1
                level_stats[level]["total"] += 1
                type_stats[subject]["total"] += 1
                total_input_tokens += token_stat["input_tokens"]
                total_output_tokens += token_stat["output_tokens"]
                running_total_tokens = total_input_tokens + total_output_tokens

                print(
                    f"[Q {total}/{len(problems)} finished] "
                    f"tokens used: input={token_stat['input_tokens']} "
                    f"output={token_stat['output_tokens']} "
                    f"question_total={token_stat['total_tokens']} "
                    f"running_total={running_total_tokens}"
                )

                token_log_fp.write(
                    json.dumps(
                        {
                            "index": total,
                            "number": problem.get("number", total),
                            "level": level,
                            "type": subject,
                            "input_tokens": token_stat["input_tokens"],
                            "output_tokens": token_stat["output_tokens"],
                            "total_tokens": token_stat["total_tokens"],
                            "running_total_tokens": running_total_tokens,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                token_log_fp.flush()

                results.append(
                    {
                        "problem": question,
                        "level": level,
                        "type": subject,
                        "gold_answer": gold_answer,
                        "predicted_answer": pred_answer,
                        "full_response": response,
                        "correct": is_correct,
                        "input_tokens": token_stat["input_tokens"],
                        "output_tokens": token_stat["output_tokens"],
                        "total_tokens": token_stat["total_tokens"],
                    }
                )
                pbar.update(1)

            idx += current_batch_size
            
            # Clear memory to prevent VRAM climbing
            if 'responses' in locals():
                del responses
            if 'batch_token_stats' in locals():
                del batch_token_stats
            torch.cuda.empty_cache()
            import gc
            gc.collect()

    pbar.close()

    accuracy = correct / total if total > 0 else 0.0

    level_accuracy = {
        level: stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
        for level, stats in level_stats.items()
    }
    type_accuracy = {
        subject: stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
        for subject, stats in type_stats.items()
    }

    eval_results = {
        "benchmark": BENCHMARK_NAME,
        "model_path": model_path,
        "lora_path": lora_path,
        "is_baseline": use_baseline,
        "timestamp": datetime.now().isoformat(),
        "total_samples": total,
        "correct": correct,
        "accuracy": accuracy,
        "accuracy_percent": f"{accuracy * 100:.2f}%",
        "max_output_tokens": max_output_tokens,
        "temperature": temperature,
        "accuracy_by_level": {k: f"{v * 100:.2f}%" for k, v in sorted(level_accuracy.items())},
        "accuracy_by_type": {k: f"{v * 100:.2f}%" for k, v in sorted(type_accuracy.items())},
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_input_tokens + total_output_tokens,
        "token_log_file": token_log_file,
        "results": results,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print(f"RESULTS: {BENCHMARK_NAME}")
    print("=" * 60)
    print(f"Total samples:    {total}")
    print(f"Correct:          {correct}")
    print(f"Overall Accuracy: {accuracy * 100:.2f}%")
    print(f"Input tokens:     {total_input_tokens}")
    print(f"Output tokens:    {total_output_tokens}")
    print(f"Total tokens:     {total_input_tokens + total_output_tokens}")

    print("\nAccuracy by Level:")
    for level, acc in sorted(level_accuracy.items()):
        print(f"  {level}: {acc * 100:.2f}%")

    print("\nAccuracy by Subject:")
    for subject, acc in sorted(type_accuracy.items()):
        print(f"  {subject}: {acc * 100:.2f}%")

    print(f"\nResults saved to: {output_file}")
    print(f"Token log saved to: {token_log_file}")

    return eval_results


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description=f"Evaluate model on {BENCHMARK_NAME}")

    parser.add_argument("--model", type=str, default="./outputs/models/final_model", help="Path to the model")
    parser.add_argument("--lora", type=str, default=None, help="Path to LoRA adapters")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum samples to evaluate")
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=None,
        help="Maximum generated tokens per question (default: auto by remaining context window)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Generation temperature (default: 1.0)",
    )
    parser.add_argument("--output", type=str, default=None, help="Output file for results")
    parser.add_argument("--baseline", action="store_true", help="Evaluate base model (no fine-tuning)")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size for generation (auto-reduced on OOM)")
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Use 4-bit quantization for lower memory and higher speed",
    )
    parser.add_argument(
        "--realtime-token-progress",
        action="store_true",
        help="Print generated token count every second for each question",
    )
    parser.add_argument("--token-log", type=str, default=None, help="JSONL file to log per-question token usage")

    args = parser.parse_args()

    evaluate_math(
        model_path=args.model,
        lora_path=args.lora,
        max_samples=args.max_samples,
        max_output_tokens=args.max_output_tokens,
        temperature=args.temperature,
        output_file=args.output,
        use_baseline=args.baseline,
        batch_size=args.batch_size,
        load_in_4bit=args.load_in_4bit,
        realtime_token_progress=args.realtime_token_progress,
        token_log_file=args.token_log,
    )


if __name__ == "__main__":
    main()
