"""
GSM8K Benchmark Evaluation Script
Evaluates model on grade school math problems (8.5K test set).
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
from typing import Optional, Tuple, List
from tqdm import tqdm
from datetime import datetime
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

BENCHMARK_NAME = "GSM8K"
DATASET_NAME = "gsm8k"
DATASET_CONFIG = "main"
SPLIT = "test"

OUTPUT_DIR = "./outputs/results"
SYSTEM_PROMPT = (
    "You are a GSM8K math solver. "
    "Solve carefully, verify arithmetic, and provide one final numeric answer only. "
    "End with: Final answer: \\boxed{number}."
)





import urllib.request
import urllib.error
import urllib.parse
import json

def count_output_tokens(a, b): return 0
def resolve_context_window(a, b): return 30000

def generate_responses_batch(
    model, # Unused
    tokenizer, # Unused
    prompts: List[str],
    context_window: Optional[int] = None,
    max_output_tokens: Optional[int] = None,
    temperature: float = 1.0,
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
            "temperature": 1.0,
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

def extract_gsm8k_answer(text: str) -> Optional[str]:
    """
    Extract numerical answer from GSM8K format.

    Args:
        text: The generated text or ground truth

    Returns:
        Extracted numerical answer as string
    """
    match = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", text)
    if match:
        return match.group(1).replace(",", "")

    boxed_match = re.search(r"\\boxed\{([^}]+)\}", text)
    if boxed_match:
        answer = boxed_match.group(1)
        num_match = re.search(r"-?[\d,]+(?:\.\d+)?", answer)
        if num_match:
            return num_match.group(0).replace(",", "")
        return answer.strip()

    numbers = re.findall(r"-?[\d,]+(?:\.\d+)?", text)
    if numbers:
        return numbers[-1].replace(",", "")

    return None


def normalize_number(answer: Optional[str]) -> Optional[float]:
    """Convert answer string to float for comparison."""
    if answer is None:
        return None

    try:
        clean = str(answer).replace(",", "").strip()
        return float(clean)
    except (ValueError, AttributeError):
        return None


def answers_match(pred: Optional[str], gold: Optional[str]) -> bool:
    """
    Check if predicted answer matches gold answer.

    Args:
        pred: Predicted answer
        gold: Gold/ground truth answer

    Returns:
        True if answers match
    """
    pred_num = normalize_number(pred)
    gold_num = normalize_number(gold)

    if pred_num is None or gold_num is None:
        return str(pred).strip() == str(gold).strip()

    return abs(pred_num - gold_num) < 1e-6


# =============================================================================
# Dataset Loading
# =============================================================================

def load_gsm8k_dataset() -> List[dict]:
    """Load GSM8K test dataset."""
    print(f"Loading {BENCHMARK_NAME} dataset...")

    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split=SPLIT)

    problems = []
    for idx, item in enumerate(dataset, start=1):
        question = item.get("question", "")
        answer_text = item.get("answer", "")
        gold_answer = extract_gsm8k_answer(answer_text)
        if gold_answer is None:
            gold_answer = str(answer_text).strip()

        if question:
            problems.append(
                {
                    "question": question,
                    "answer": answer_text,
                    "gold_answer": gold_answer,
                    "number": idx,
                }
            )

    print(f"Loaded {len(problems)} problems")
    return problems


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_gsm8k(
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
    Evaluate model on GSM8K benchmark.

    Args:
        model_path: Path to the model
        lora_path: Path to LoRA adapters (optional)
        max_samples: Maximum samples to evaluate (None = all)
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

    problems = load_gsm8k_dataset()

    if max_samples:
        problems = problems[:max_samples]
        print(f"\nEvaluating on {max_samples} samples")

    if output_file is None:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        suffix = "_baseline" if use_baseline else "_finetuned"
        output_file = os.path.join(OUTPUT_DIR, f"gsm8k{suffix}_results.json")

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

    print("\nRunning evaluation...")
    pbar = tqdm(total=len(problems), desc=BENCHMARK_NAME)
    idx = 0

    with open(token_log_file, "w", encoding="utf-8") as token_log_fp:
        while idx < len(problems):
            current_batch_size = min(batch_size, len(problems) - idx)
            batch = problems[idx : idx + current_batch_size]
            questions = [item["question"] for item in batch]

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
                question = problem["question"]
                gold_answer = problem["gold_answer"]

                pred_answer = extract_gsm8k_answer(response)
                is_correct = answers_match(pred_answer, gold_answer)

                if is_correct:
                    correct += 1

                total += 1
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
                            "input_tokens": token_stat["input_tokens"],
                            "output_tokens": token_stat["output_tokens"],
                            "total_tokens": token_stat["total_tokens"],
                            "running_total_tokens": running_total_tokens,
                            "gold_answer": gold_answer,
                            "predicted_answer": pred_answer,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                token_log_fp.flush()

                results.append(
                    {
                        "question": question,
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
    print(f"Total samples:  {total}")
    print(f"Correct:        {correct}")
    print(f"Accuracy:       {accuracy * 100:.2f}%")
    print(f"Input tokens:   {total_input_tokens}")
    print(f"Output tokens:  {total_output_tokens}")
    print(f"Total tokens:   {total_input_tokens + total_output_tokens}")
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

    evaluate_gsm8k(
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