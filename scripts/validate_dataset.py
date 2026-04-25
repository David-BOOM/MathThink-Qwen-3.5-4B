"""
Dataset Validation Script
Validates the processed dataset for training readiness.
"""

import json
import os
from pathlib import Path
from collections import Counter, defaultdict
from tqdm import tqdm
import random


PROCESSED_FILE = "./Nemotron-Math-v3/train_processed.jsonl"
VALIDATION_REPORT = "./Nemotron-Math-v3/validation_report.json"


def count_lines(filepath: str) -> int:
    """Count total lines in file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return sum(1 for _ in f)


def estimate_tokens(text: str) -> int:
    """Rough token estimate (words * 1.3 for English/math mix)."""
    return int(len(text.split()) * 1.3)


def validate_dataset():
    """Validate the processed dataset."""
    
    if not os.path.exists(PROCESSED_FILE):
        print(f"Error: Processed file not found: {PROCESSED_FILE}")
        print("Run preprocess_dataset.py first.")
        return
    
    print(f"Validating: {PROCESSED_FILE}")
    total_lines = count_lines(PROCESSED_FILE)
    print(f"Total entries: {total_lines:,}\n")
    
    # Validation metrics
    issues = {
        "empty_messages": 0,
        "missing_user": 0,
        "missing_assistant": 0,
        "wrong_ending": 0,
        "empty_content": 0,
        "consecutive_same_role": 0,
    }
    
    # Statistics
    role_counts = Counter()
    message_lengths = defaultdict(list)
    conversation_lengths = []
    token_estimates = []
    
    # Sample conversations for manual review
    samples = []
    sample_indices = set(random.sample(range(total_lines), min(10, total_lines)))
    
    with open(PROCESSED_FILE, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(tqdm(f, total=total_lines, desc="Validating")):
            line = line.strip()
            if not line:
                continue
            
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                issues["empty_messages"] += 1
                continue
            
            messages = entry.get("messages", [])
            
            if not messages:
                issues["empty_messages"] += 1
                continue
            
            # Track conversation length
            conversation_lengths.append(len(messages))
            
            # Validate structure
            has_user = False
            has_assistant = False
            prev_role = None
            total_tokens = 0
            
            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                
                role_counts[role] += 1
                
                if role == "user":
                    has_user = True
                elif role == "assistant":
                    has_assistant = True
                
                if not content or not content.strip():
                    issues["empty_content"] += 1
                
                if role == prev_role:
                    issues["consecutive_same_role"] += 1
                
                prev_role = role
                
                # Track message lengths
                char_len = len(content)
                message_lengths[role].append(char_len)
                total_tokens += estimate_tokens(content)
            
            token_estimates.append(total_tokens)
            
            if not has_user:
                issues["missing_user"] += 1
            if not has_assistant:
                issues["missing_assistant"] += 1
            if messages[-1].get("role") != "assistant":
                issues["wrong_ending"] += 1
            
            # Collect samples
            if idx in sample_indices:
                samples.append({
                    "index": idx,
                    "num_messages": len(messages),
                    "roles": [m["role"] for m in messages],
                    "preview": {
                        "user": messages[0]["content"][:200] + "..." if messages else "",
                        "assistant": messages[-1]["content"][:200] + "..." if messages else ""
                    }
                })
    
    # Calculate statistics
    def calc_stats(values):
        if not values:
            return {"min": 0, "max": 0, "mean": 0, "median": 0}
        sorted_vals = sorted(values)
        return {
            "min": min(values),
            "max": max(values),
            "mean": sum(values) / len(values),
            "median": sorted_vals[len(sorted_vals) // 2]
        }
    
    report = {
        "total_entries": total_lines,
        "valid_entries": total_lines - sum(issues.values()),
        "issues": issues,
        "role_distribution": dict(role_counts),
        "conversation_length_stats": calc_stats(conversation_lengths),
        "token_estimate_stats": calc_stats(token_estimates),
        "message_length_stats": {
            role: calc_stats(lengths) 
            for role, lengths in message_lengths.items()
        },
        "samples": samples
    }
    
    # Save report
    with open(VALIDATION_REPORT, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("VALIDATION REPORT")
    print("="*60)
    print(f"Total entries:    {total_lines:,}")
    print(f"Valid entries:    {report['valid_entries']:,}")
    print(f"Validity rate:    {report['valid_entries']/total_lines*100:.2f}%")
    
    print("\nIssues found:")
    for issue, count in issues.items():
        status = "✓" if count == 0 else "⚠"
        print(f"  {status} {issue}: {count:,}")
    
    print("\nRole distribution:")
    for role, count in role_counts.items():
        print(f"  - {role}: {count:,}")
    
    print("\nConversation length (messages):")
    stats = report["conversation_length_stats"]
    print(f"  Min: {stats['min']}, Max: {stats['max']}, Mean: {stats['mean']:.1f}, Median: {stats['median']}")
    
    print("\nEstimated tokens per conversation:")
    stats = report["token_estimate_stats"]
    print(f"  Min: {stats['min']}, Max: {stats['max']}, Mean: {stats['mean']:.1f}, Median: {stats['median']}")
    
    print(f"\nFull report saved to: {VALIDATION_REPORT}")
    
    # Show sample
    if samples:
        print("\n" + "="*60)
        print("SAMPLE CONVERSATION")
        print("="*60)
        sample = samples[0]
        print(f"Index: {sample['index']}")
        print(f"Roles: {' → '.join(sample['roles'])}")
        print(f"\nUser: {sample['preview']['user']}")
        print(f"\nAssistant: {sample['preview']['assistant']}")
    
    return report


if __name__ == "__main__":
    validate_dataset()
