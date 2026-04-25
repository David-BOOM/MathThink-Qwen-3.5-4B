"""
Unified Evaluation Runner
Runs selected benchmark evaluations and aggregates results.
Also supports benchmark dataset pre-download.
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional, List

EVAL_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(EVAL_DIR, ".."))

if EVAL_DIR not in sys.path:
    sys.path.insert(0, EVAL_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from eval_gsm8k_lmstudio import evaluate_gsm8k, load_gsm8k_dataset
from eval_math_lmstudio import evaluate_math, load_math_dataset
from eval_aime_lmstudio import evaluate_aime, load_aime_dataset


# =============================================================================
# Configuration
# =============================================================================

OUTPUT_DIR = "./outputs/results"
BENCHMARKS = ["gsm8k", "math", "aime2025"]


# =============================================================================
# Dataset Download
# =============================================================================

def download_required_datasets(benchmarks: Optional[List[str]] = None) -> dict:
    """Download required datasets by materializing benchmark loaders once."""
    if benchmarks is None:
        benchmarks = BENCHMARKS

    print("=" * 60)
    print("DATASET DOWNLOAD")
    print("=" * 60)
    print(f"Benchmarks: {', '.join(benchmarks)}")
    print()

    summary = {
        "timestamp": datetime.now().isoformat(),
        "benchmarks": {},
    }

    for benchmark in benchmarks:
        try:
            normalized_benchmark = "aime2025" if benchmark == "aime" else benchmark
            print(f"Downloading dataset for: {benchmark.upper()}")
            if normalized_benchmark == "gsm8k":
                problems = load_gsm8k_dataset()
            elif normalized_benchmark == "math":
                problems = load_math_dataset()
            elif normalized_benchmark == "aime2025":
                problems = load_aime_dataset()
            else:
                print(f"Unknown benchmark: {benchmark}")
                summary["benchmarks"][benchmark] = {
                    "status": "error",
                    "error": f"Unknown benchmark: {benchmark}",
                }
                continue

            summary["benchmarks"][benchmark] = {
                "status": "ok",
                "num_samples": len(problems),
            }
            print(f"  Downloaded and loaded {len(problems)} samples")
        except Exception as exc:
            summary["benchmarks"][benchmark] = {
                "status": "error",
                "error": str(exc),
            }
            print(f"  Failed: {exc}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    summary_file = os.path.join(OUTPUT_DIR, "dataset_download_summary.json")
    with open(summary_file, "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2, ensure_ascii=False)

    print()
    print(f"Dataset summary saved to: {summary_file}")
    return summary


# =============================================================================
# Main Runner
# =============================================================================

def run_all_evaluations(
    model_path: str,
    lora_path: Optional[str] = None,
    max_samples: Optional[int] = None,
    max_output_tokens: Optional[int] = None,
    temperature: float = 0.0,
    benchmarks: Optional[List[str]] = None,
    use_baseline: bool = False,
    batch_size: int = 2,
    load_in_4bit: bool = False,
    realtime_token_progress: bool = False,
    token_log_dir: Optional[str] = None,
    download_datasets: bool = False,
    download_only: bool = False,
):
    """
    Run selected benchmark evaluations.

    Args:
        model_path: Path to model
        lora_path: Path to LoRA adapters
        max_samples: Maximum samples per benchmark
        max_output_tokens: Maximum generated tokens per question (None = auto)
        temperature: Generation temperature (0.0 = greedy deterministic)
        benchmarks: Benchmarks to run
        use_baseline: Evaluate base model without LoRA
        batch_size: Generation batch size
        load_in_4bit: Enable 4-bit loading
        realtime_token_progress: Enable per-second token progress (forces batch=1)
        token_log_dir: Directory for per-benchmark token logs
        download_datasets: Pre-download benchmark datasets before evaluation
        download_only: Download datasets only and skip all evaluation

    Returns:
        Aggregated results dictionary
    """
    if benchmarks is None:
        benchmarks = BENCHMARKS

    if max_output_tokens is not None and max_output_tokens < 1:
        raise ValueError("max_output_tokens must be >= 1 when provided")
    if temperature < 0:
        raise ValueError("temperature must be >= 0")

    dataset_download_summary = None
    if download_datasets or download_only:
        dataset_download_summary = download_required_datasets(benchmarks=benchmarks)

    if download_only:
        return {
            "model_path": model_path,
            "lora_path": lora_path,
            "is_baseline": use_baseline,
            "max_output_tokens": max_output_tokens,
            "temperature": temperature,
            "timestamp": datetime.now().isoformat(),
            "download_only": True,
            "dataset_download": dataset_download_summary,
            "benchmarks": {},
        }

    print("=" * 60)
    print("UNIFIED EVALUATION RUNNER")
    print("=" * 60)
    print(f"Model: {model_path}")
    if lora_path:
        print(f"LoRA: {lora_path}")
    print(f"Baseline mode: {use_baseline}")
    print(f"Benchmarks: {', '.join(benchmarks)}")
    if max_samples:
        print(f"Max samples per benchmark: {max_samples}")
    if max_output_tokens is None:
        print("Max output tokens per question: auto (up to remaining context window)")
    else:
        print(f"Max output tokens per question: {max_output_tokens}")
    print(f"Batch size: {batch_size}")
    print(f"Temperature: {temperature}")
    print(f"4-bit quantization: {load_in_4bit}")
    print(f"Realtime token progress: {realtime_token_progress}")
    if token_log_dir:
        print(f"Token log dir: {token_log_dir}")
    print(f"Started at: {datetime.now().isoformat()}")
    print()

    all_results = {
        "model_path": model_path,
        "lora_path": lora_path,
        "is_baseline": use_baseline,
        "max_output_tokens": max_output_tokens,
        "temperature": temperature,
        "timestamp": datetime.now().isoformat(),
        "dataset_download": dataset_download_summary,
        "benchmarks": {},
    }

    if token_log_dir:
        Path(token_log_dir).mkdir(parents=True, exist_ok=True)

    suffix = "_baseline" if use_baseline else "_finetuned"

    for benchmark in benchmarks:
        normalized_benchmark = "aime2025" if benchmark == "aime" else benchmark
        print("\n" + "=" * 60)
        print(f"Running: {benchmark.upper()}")
        print("=" * 60 + "\n")

        output_file = os.path.join(OUTPUT_DIR, f"{normalized_benchmark}{suffix}_results.json")
        token_log_file = None
        if token_log_dir:
            token_log_file = os.path.join(token_log_dir, f"{normalized_benchmark}{suffix}_token_log.jsonl")

        try:
            if normalized_benchmark == "gsm8k":
                results = evaluate_gsm8k(
                    model_path=model_path,
                    lora_path=lora_path,
                    max_samples=max_samples,
                    max_output_tokens=max_output_tokens,
                    temperature=temperature,
                    output_file=output_file,
                    use_baseline=use_baseline,
                    batch_size=batch_size,
                    load_in_4bit=load_in_4bit,
                    realtime_token_progress=realtime_token_progress,
                    token_log_file=token_log_file,
                )
            elif normalized_benchmark == "math":
                results = evaluate_math(
                    model_path=model_path,
                    lora_path=lora_path,
                    max_samples=max_samples,
                    max_output_tokens=max_output_tokens,
                    temperature=temperature,
                    output_file=output_file,
                    use_baseline=use_baseline,
                    batch_size=batch_size,
                    load_in_4bit=load_in_4bit,
                    realtime_token_progress=realtime_token_progress,
                    token_log_file=token_log_file,
                )
            elif normalized_benchmark == "aime2025":
                results = evaluate_aime(
                    model_path=model_path,
                    lora_path=lora_path,
                    max_samples=max_samples,
                    max_output_tokens=max_output_tokens,
                    temperature=temperature,
                    output_file=output_file,
                    use_baseline=use_baseline,
                    batch_size=batch_size,
                    load_in_4bit=load_in_4bit,
                    realtime_token_progress=realtime_token_progress,
                    token_log_file=token_log_file,
                )
            else:
                print(f"Unknown benchmark: {benchmark}")
                continue

            all_results["benchmarks"][normalized_benchmark] = {
                "total_samples": results["total_samples"],
                "correct": results["correct"],
                "accuracy": results["accuracy"],
                "accuracy_percent": results["accuracy_percent"],
                "total_input_tokens": results.get("total_input_tokens", 0),
                "total_output_tokens": results.get("total_output_tokens", 0),
                "total_tokens": results.get("total_tokens", 0),
                "token_log_file": results.get("token_log_file"),
            }

            if "accuracy_by_level" in results:
                all_results["benchmarks"][normalized_benchmark]["accuracy_by_level"] = results["accuracy_by_level"]
            if "accuracy_by_type" in results:
                all_results["benchmarks"][normalized_benchmark]["accuracy_by_type"] = results["accuracy_by_type"]
            if "accuracy_by_year" in results:
                all_results["benchmarks"][normalized_benchmark]["accuracy_by_year"] = results["accuracy_by_year"]

        except Exception as exc:
            print(f"Error running {benchmark}: {exc}")
            import traceback

            traceback.print_exc()
            all_results["benchmarks"][benchmark] = {"error": str(exc)}

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    summary_file = os.path.join(OUTPUT_DIR, f"all_results{suffix}.json")

    with open(summary_file, "w", encoding="utf-8") as fp:
        json.dump(all_results, fp, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Mode: {'Baseline' if use_baseline else 'Fine-tuned'}")
    print()

    print(f"{'Benchmark':<12} {'Accuracy':>10} {'Correct':>10} {'Total':>10}")
    print("-" * 45)

    for benchmark, results in all_results["benchmarks"].items():
        if "error" in results:
            print(f"{benchmark.upper():<12} {'ERROR':>10}")
        else:
            print(
                f"{benchmark.upper():<12} "
                f"{results['accuracy_percent']:>10} "
                f"{results['correct']:>10} "
                f"{results['total_samples']:>10}"
            )

    print()
    print(f"Summary saved to: {summary_file}")
    print(f"Completed at: {datetime.now().isoformat()}")

    return all_results


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run benchmark evaluations and dataset downloads")

    parser.add_argument("--model", type=str, default="./outputs/models/final_model", help="Path to the model")
    parser.add_argument("--lora", type=str, default=None, help="Path to LoRA adapters")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum samples per benchmark")
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
    parser.add_argument(
        "--benchmarks",
        type=str,
        nargs="+",
        choices=BENCHMARKS + ["aime"],
        default=None,
        help="Benchmarks to run (default: all; 'aime' is accepted as alias for 'aime2025')",
    )
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
    parser.add_argument(
        "--token-log-dir",
        type=str,
        default=None,
        help="Directory to store benchmark token JSONL logs",
    )
    parser.add_argument(
        "--download-datasets",
        action="store_true",
        help="Download benchmark datasets before running evaluations",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Download benchmark datasets only and skip evaluation",
    )

    args = parser.parse_args()

    run_all_evaluations(
        model_path=args.model,
        lora_path=args.lora,
        max_samples=args.max_samples,
        max_output_tokens=args.max_output_tokens,
        temperature=args.temperature,
        benchmarks=args.benchmarks,
        use_baseline=args.baseline,
        batch_size=args.batch_size,
        load_in_4bit=args.load_in_4bit,
        realtime_token_progress=args.realtime_token_progress,
        token_log_dir=args.token_log_dir,
        download_datasets=args.download_datasets,
        download_only=args.download_only,
    )


if __name__ == "__main__":
    main()