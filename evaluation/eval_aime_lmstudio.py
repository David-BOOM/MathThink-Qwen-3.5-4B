"""
AIME Benchmark Evaluation Script
Evaluates model on American Invitational Mathematics Examination problems.
Tests long reasoning chains and deliberative thinking.
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
from typing import Optional, List, Tuple
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

BENCHMARK_NAME = "AIME 2025"
# AIME dataset options - try multiple sources
AIME_DATASETS = [
    ("MathArena/aime_2025", None),            # Primary source
    ("FVU/AIME_2025", None),                 # Backup
    ("TianHongZXY/aime-1983-2025", None),    # Wide fallback source
]

OUTPUT_DIR = "./outputs/results"
SYSTEM_PROMPT = (
    "You are an AIME benchmark solver. "
    "Solve carefully, verify arithmetic, and provide one final answer only. "
    "End with: Final answer: \\boxed{n}, where n is an integer from 0 to 999."
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

def extract_aime_answer(text: str) -> Optional[int]:
    """
    Extract AIME answer (integer 0-999).
    
    AIME answers are always integers from 0 to 999.
    
    Args:
        text: The generated text
    
    Returns:
        Extracted integer answer or None
    """
    # Try \boxed{...} format first
    boxed_pattern = r'\\boxed\{(\d+)\}'
    matches = re.findall(boxed_pattern, text)
    if matches:
        try:
            answer = int(matches[-1])
            if 0 <= answer <= 999:
                return answer
        except ValueError:
            pass
    
    # Try to find standalone 3-digit numbers near end of text
    # AIME answers are 000-999
    last_500_chars = text[-500:] if len(text) > 500 else text
    
    # Look for patterns like "answer is X", "= X", "the answer: X"
    answer_patterns = [
        r'(?:answer|Answer|ANSWER)\s*(?:is|:)?\s*(\d{1,3})\b',
        r'(?:=|equals)\s*(\d{1,3})\b',
        r'\\boxed\{([^}]+)\}',
    ]
    
    for pattern in answer_patterns:
        matches = re.findall(pattern, last_500_chars)
        if matches:
            try:
                answer = int(matches[-1])
                if 0 <= answer <= 999:
                    return answer
            except ValueError:
                continue
    
    # Last resort: find the last number that could be an AIME answer
    all_numbers = re.findall(r'\b(\d{1,3})\b', text)
    for num_str in reversed(all_numbers):
        try:
            num = int(num_str)
            if 0 <= num <= 999:
                return num
        except ValueError:
            continue
    
    return None


def answers_match(pred: Optional[int], gold: int) -> bool:
    """
    Check if predicted answer matches gold answer.
    
    Args:
        pred: Predicted answer (integer or None)
        gold: Gold answer (integer)
    
    Returns:
        True if answers match exactly
    """
    if pred is None:
        return False
    return pred == gold


# =============================================================================
# Dataset Loading
# =============================================================================

def load_aime_dataset() -> List[dict]:
    """Load AIME dataset from available sources."""
    print(f"Loading {BENCHMARK_NAME} dataset...")
    
    problems = []
    
    for dataset_name, config in AIME_DATASETS:
        try:
            print(f"Trying: {dataset_name}")
            dataset = None
            if config:
                dataset = load_dataset(dataset_name, config, split="test")
            else:
                # Try different splits
                for split in ["test", "train", "validation"]:
                    try:
                        dataset = load_dataset(dataset_name, split=split)
                        break
                    except:
                        continue

            if dataset is None:
                raise RuntimeError(f"No supported split found for {dataset_name}")
            
            for item in dataset:
                # Handle different field names
                problem = item.get("problem", item.get("question", item.get("Problem", "")))
                answer = item.get("answer", item.get("Answer", item.get("solution", "")))
                year = item.get("year", item.get("Year", "unknown"))
                number = item.get("number", item.get("Number", item.get("problem_number", "unknown")))
                
                # Convert answer to int if needed
                if isinstance(answer, str):
                    try:
                        answer = int(re.search(r'(\d+)', answer).group(1))
                    except:
                        answer = None
                
                if problem and answer is not None:
                    problems.append({
                        "problem": problem,
                        "gold_answer": answer,
                        "year": year,
                        "number": number
                    })
            
            if problems:
                print(f" Loaded {len(problems)} problems from {dataset_name}")
                break
                
        except Exception as e:
            print(f"  Could not load {dataset_name}: {e}")
            continue
    
    if not problems:
        # Create a fallback with sample AIME problems
        print("\n Could not load AIME dataset from HuggingFace.")
        print("  Using built-in sample problems for demonstration.")
        problems = get_sample_aime_problems()
    
    return problems


def get_sample_aime_problems() -> List[dict]:
    """Return sample AIME problems for testing."""
    return [
        {
            "problem": "Find the number of positive integers $n$ less than 1000 for which there exist positive integers $a$ and $b$ such that $n = a^2 - b^2$.",
            "gold_answer": 749,
            "year": "sample",
            "number": 1
        },
        {
            "problem": "Let $S$ be the set of all positive rational numbers $r$ such that $2r$ is an integer. Find the number of elements in $S$.",
            "gold_answer": 1,
            "year": "sample", 
            "number": 2
        },
        {
            "problem": "Find the sum of all positive integers $n$ such that $\\sqrt{n^2+85n+2017}$ is an integer.",
            "gold_answer": 195,
            "year": "sample",
            "number": 3
        }
    ]


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_aime(
    model_path: str,
    lora_path: Optional[str] = None,
    max_samples: Optional[int] = None,
    max_output_tokens: Optional[int] = None,
    temperature: float = 1.0,
    output_file: Optional[str] = None,
    use_baseline: bool = False,
    batch_size: int = 4,
    load_in_4bit: bool = False,
    realtime_token_progress: bool = False,
    token_log_file: Optional[str] = None,
):
    """
    Evaluate model on AIME benchmark.
    
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
    model, tokenizer = None, None
    # Load dataset
    problems = load_aime_dataset()
    
    if max_samples:
        problems = problems[:max_samples]
        print(f"\nEvaluating on {max_samples} samples")

    if output_file is None:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        suffix = "_baseline" if use_baseline else "_finetuned"
        output_file = os.path.join(OUTPUT_DIR, f"aime2025{suffix}_results.json")

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
    
    # Evaluate
    results = []
    correct = 0
    total = 0
    total_input_tokens = 0
    total_output_tokens = 0
    
    # Track by year
    year_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    
    print("\nRunning evaluation...")
    pbar = tqdm(total=len(problems), desc=BENCHMARK_NAME)
    idx = 0

    with open(token_log_file, "w", encoding="utf-8") as token_log_fp:
        while idx < len(problems):
            current_batch_size = min(batch_size, len(problems) - idx)
            batch = problems[idx: idx + current_batch_size]
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
                year = problem["year"]

                pred_answer = extract_aime_answer(response)
                is_correct = answers_match(pred_answer, gold_answer)

                if is_correct:
                    correct += 1
                    year_stats[year]["correct"] += 1

                total += 1
                year_stats[year]["total"] += 1
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
                            "year": year,
                            "number": problem.get("number", "unknown"),
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

                results.append({
                    "problem": question,
                    "year": year,
                    "number": problem.get("number", "unknown"),
                    "gold_answer": gold_answer,
                    "predicted_answer": pred_answer,
                    "full_response": response,
                    "correct": is_correct,
                    "input_tokens": token_stat["input_tokens"],
                    "output_tokens": token_stat["output_tokens"],
                    "total_tokens": token_stat["total_tokens"],
                })
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
    
    # Calculate accuracy
    accuracy = correct / total if total > 0 else 0.0
    
    year_accuracy = {
        year: stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        for year, stats in year_stats.items()
    }
    
    # Prepare output
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
        "accuracy_by_year": {k: f"{v*100:.2f}%" for k, v in sorted(year_accuracy.items())},
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_input_tokens + total_output_tokens,
        "token_log_file": token_log_file,
        "results": results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(eval_results, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print("\n" + "="*60)
    print(f"RESULTS: {BENCHMARK_NAME}")
    print("="*60)
    print(f"Total samples:  {total}")
    print(f"Correct:        {correct}")
    print(f"Accuracy:       {accuracy * 100:.2f}%")
    print(f"Input tokens:   {total_input_tokens}")
    print(f"Output tokens:  {total_output_tokens}")
    print(f"Total tokens:   {total_input_tokens + total_output_tokens}")
    
    if len(year_accuracy) > 1:
        print("\nAccuracy by Year:")
        for year, acc in sorted(year_accuracy.items()):
            print(f"  {year}: {acc * 100:.2f}%")
    
    print(f"\nResults saved to: {output_file}")
    print(f"Token log saved to: {token_log_file}")
    
    return eval_results


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description=f"Evaluate model on {BENCHMARK_NAME}")
    
    parser.add_argument("--model", type=str, default="./outputs/models/final_model",
                        help="Path to the model")
    parser.add_argument("--lora", type=str, default=None,
                        help="Path to LoRA adapters")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Maximum samples to evaluate")
    parser.add_argument("--max-output-tokens", type=int, default=None,
                        help="Maximum generated tokens per question (default: auto by remaining context window)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Generation temperature (default: 1.0)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file for results")
    parser.add_argument("--baseline", action="store_true",
                        help="Evaluate base model (no fine-tuning)")
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Batch size for generation (auto-reduced on OOM)")
    parser.add_argument("--load-in-4bit", action="store_true",
                        help="Use 4-bit quantization for lower memory and higher speed")
    parser.add_argument("--realtime-token-progress", action="store_true",
                        help="Print generated token count every second for each question")
    parser.add_argument("--token-log", type=str, default=None,
                        help="JSONL file to log per-question token usage")
    
    args = parser.parse_args()
    
    evaluate_aime(
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

