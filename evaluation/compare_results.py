"""
Results Comparison Script
Compares baseline vs fine-tuned model performance.
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path


# =============================================================================
# Configuration
# =============================================================================

OUTPUT_DIR = "./outputs/results"
BENCHMARKS = ["gsm8k", "math", "aime"]


# =============================================================================
# Comparison Functions
# =============================================================================

def load_results(filepath: str) -> dict:
    """Load results from JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def compare_results(baseline_file: str, finetuned_file: str, output_file: str = None):
    """
    Compare baseline and fine-tuned results.
    
    Args:
        baseline_file: Path to baseline results JSON
        finetuned_file: Path to fine-tuned results JSON
        output_file: Where to save comparison report
    """
    
    print("="*60)
    print("RESULTS COMPARISON")
    print("="*60)
    
    # Load results
    print(f"\nLoading baseline: {baseline_file}")
    baseline = load_results(baseline_file)
    
    print(f"Loading fine-tuned: {finetuned_file}")
    finetuned = load_results(finetuned_file)
    
    # Build comparison
    comparison = {
        "timestamp": datetime.now().isoformat(),
        "baseline_file": baseline_file,
        "finetuned_file": finetuned_file,
        "baseline_model": baseline.get("model_path", "unknown"),
        "finetuned_model": finetuned.get("model_path", "unknown"),
        "benchmarks": {}
    }
    
    # Compare each benchmark
    all_benchmarks = set(baseline.get("benchmarks", {}).keys()) | set(finetuned.get("benchmarks", {}).keys())
    
    for benchmark in sorted(all_benchmarks):
        base_results = baseline.get("benchmarks", {}).get(benchmark, {})
        ft_results = finetuned.get("benchmarks", {}).get(benchmark, {})
        
        if "error" in base_results or "error" in ft_results:
            comparison["benchmarks"][benchmark] = {
                "baseline_error": base_results.get("error"),
                "finetuned_error": ft_results.get("error")
            }
            continue
        
        base_acc = base_results.get("accuracy", 0)
        ft_acc = ft_results.get("accuracy", 0)
        
        improvement = ft_acc - base_acc
        improvement_pct = (improvement / base_acc * 100) if base_acc > 0 else 0
        
        comparison["benchmarks"][benchmark] = {
            "baseline_accuracy": base_acc,
            "baseline_accuracy_pct": f"{base_acc * 100:.2f}%",
            "finetuned_accuracy": ft_acc,
            "finetuned_accuracy_pct": f"{ft_acc * 100:.2f}%",
            "absolute_improvement": improvement,
            "absolute_improvement_pct": f"{improvement * 100:+.2f}%",
            "relative_improvement": f"{improvement_pct:+.2f}%",
            "baseline_correct": base_results.get("correct", 0),
            "finetuned_correct": ft_results.get("correct", 0),
            "total_samples": ft_results.get("total_samples", base_results.get("total_samples", 0))
        }
    
    # Calculate overall improvement
    total_base_correct = sum(
        comparison["benchmarks"][b].get("baseline_correct", 0) 
        for b in comparison["benchmarks"] 
        if "baseline_correct" in comparison["benchmarks"][b]
    )
    total_ft_correct = sum(
        comparison["benchmarks"][b].get("finetuned_correct", 0) 
        for b in comparison["benchmarks"]
        if "finetuned_correct" in comparison["benchmarks"][b]
    )
    total_samples = sum(
        comparison["benchmarks"][b].get("total_samples", 0) 
        for b in comparison["benchmarks"]
        if "total_samples" in comparison["benchmarks"][b]
    )
    
    if total_samples > 0:
        comparison["overall"] = {
            "baseline_correct": total_base_correct,
            "finetuned_correct": total_ft_correct,
            "total_samples": total_samples,
            "baseline_accuracy": total_base_correct / total_samples,
            "finetuned_accuracy": total_ft_correct / total_samples,
            "absolute_improvement": (total_ft_correct - total_base_correct) / total_samples
        }
    
    # Save comparison
    if output_file is None:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_file = os.path.join(OUTPUT_DIR, "comparison_report.json")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(comparison, f, indent=2)
    
    # Print comparison table
    print("\n" + "="*70)
    print("COMPARISON RESULTS")
    print("="*70)
    print()
    print(f"{'Benchmark':<12} {'Baseline':>12} {'Fine-tuned':>12} {'Improvement':>14} {'Status':>10}")
    print("-"*70)
    
    for benchmark in sorted(comparison["benchmarks"].keys()):
        results = comparison["benchmarks"][benchmark]
        
        if "baseline_error" in results or "finetuned_error" in results:
            print(f"{benchmark.upper():<12} {'ERROR':>12} {'ERROR':>12} {'-':>14} {'':>10}")
            continue
        
        base_pct = results["baseline_accuracy_pct"]
        ft_pct = results["finetuned_accuracy_pct"]
        impr = results["absolute_improvement_pct"]
        
        # Determine status
        if results["absolute_improvement"] > 0.01:
            status = " Better"
        elif results["absolute_improvement"] < -0.01:
            status = " Worse"
        else:
            status = "~ Same"
        
        print(f"{benchmark.upper():<12} {base_pct:>12} {ft_pct:>12} {impr:>14} {status:>10}")
    
    # Print overall
    if "overall" in comparison:
        print("-"*70)
        overall = comparison["overall"]
        base_pct = f"{overall['baseline_accuracy'] * 100:.2f}%"
        ft_pct = f"{overall['finetuned_accuracy'] * 100:.2f}%"
        impr = f"{overall['absolute_improvement'] * 100:+.2f}%"
        status = " Better" if overall["absolute_improvement"] > 0 else " Worse" if overall["absolute_improvement"] < 0 else "~ Same"
        print(f"{'OVERALL':<12} {base_pct:>12} {ft_pct:>12} {impr:>14} {status:>10}")
    
    print()
    print(f"Comparison report saved to: {output_file}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    improved = sum(1 for b, r in comparison["benchmarks"].items() 
                   if r.get("absolute_improvement", 0) > 0.01)
    degraded = sum(1 for b, r in comparison["benchmarks"].items() 
                   if r.get("absolute_improvement", 0) < -0.01)
    unchanged = len(comparison["benchmarks"]) - improved - degraded
    
    print(f"Benchmarks improved: {improved}")
    print(f"Benchmarks degraded: {degraded}")
    print(f"Benchmarks unchanged: {unchanged}")
    
    if "overall" in comparison:
        if comparison["overall"]["absolute_improvement"] > 0:
            print(f"\n Fine-tuning IMPROVED overall performance by {comparison['overall']['absolute_improvement']*100:.2f}%")
        elif comparison["overall"]["absolute_improvement"] < 0:
            print(f"\n Fine-tuning DEGRADED overall performance by {abs(comparison['overall']['absolute_improvement'])*100:.2f}%")
        else:
            print(f"\n Fine-tuning had NO SIGNIFICANT EFFECT on overall performance")
    
    return comparison


def auto_compare():
    """Automatically compare baseline and finetuned results if both exist."""
    
    baseline_file = os.path.join(OUTPUT_DIR, "all_results_baseline.json")
    finetuned_file = os.path.join(OUTPUT_DIR, "all_results_finetuned.json")
    
    if not os.path.exists(baseline_file):
        print(f"Baseline results not found: {baseline_file}")
        print("Run: python evaluation/run_all_evals.py --baseline")
        return None
    
    if not os.path.exists(finetuned_file):
        print(f"Fine-tuned results not found: {finetuned_file}")
        print("Run: python evaluation/run_all_evals.py")
        return None
    
    return compare_results(baseline_file, finetuned_file)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Compare baseline vs fine-tuned results")
    
    parser.add_argument("--baseline", type=str, default=None,
                        help="Path to baseline results JSON")
    parser.add_argument("--finetuned", type=str, default=None,
                        help="Path to fine-tuned results JSON")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file for comparison report")
    parser.add_argument("--auto", action="store_true",
                        help="Automatically find and compare results")
    
    args = parser.parse_args()
    
    if args.auto or (args.baseline is None and args.finetuned is None):
        auto_compare()
    elif args.baseline and args.finetuned:
        compare_results(args.baseline, args.finetuned, args.output)
    else:
        print("Please provide both --baseline and --finetuned, or use --auto")
        parser.print_help()


if __name__ == "__main__":
    main()

