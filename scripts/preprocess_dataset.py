"""
Dataset Preprocessing Script for Qwen 3.5 4B Fine-Tuning
Filters train_messages_only.jsonl to user→assistant pairs only.
Removes system and tool messages for pure reasoning transfer.
"""

import json
import os
from pathlib import Path
from tqdm import tqdm
from typing import Generator, Dict, List, Any


INPUT_FILE = "./Nemotron-Math-v3/train.jsonl"
OUTPUT_FILE = "./Nemotron-Math-v3/train_processed.jsonl"
STATS_FILE = "./Nemotron-Math-v3/preprocessing_stats.json"


def count_lines(filepath: str) -> int:
    """Count total lines in file for progress bar."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return sum(1 for _ in f)


def stream_jsonl(filepath: str) -> Generator[Dict[str, Any], None, None]:
    """Stream JSONL file line by line."""
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping invalid JSON: {e}")
                    continue


def filter_to_user_assistant(messages: List[Dict]) -> List[Dict]:
    """
    Filter messages to keep only user and assistant roles.
    Returns list of messages with alternating user/assistant pattern.
    """
    filtered = []
    
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        
        # Only keep user and assistant messages
        if role in ("user", "assistant"):
            # Skip empty content
            if content and content.strip():
                filtered.append({
                    "role": role,
                    "content": content
                })
    
    return filtered


def validate_conversation(messages: List[Dict]) -> bool:
    """
    Validate that conversation has proper structure:
    - At least one user and one assistant message
    - Ends with assistant (we're training on assistant responses)
    """
    if len(messages) < 2:
        return False
    
    has_user = any(m["role"] == "user" for m in messages)
    has_assistant = any(m["role"] == "assistant" for m in messages)
    
    # Should end with assistant response (what we're training)
    ends_with_assistant = messages[-1]["role"] == "assistant"
    
    return has_user and has_assistant and ends_with_assistant


def process_dataset():
    """Main processing function."""
    
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file not found: {INPUT_FILE}")
        return
    
    print(f"Counting lines in {INPUT_FILE}...")
    total_lines = count_lines(INPUT_FILE)
    print(f"Total entries: {total_lines:,}")
    
    # Statistics tracking
    stats = {
        "total_input": 0,
        "total_output": 0,
        "skipped_empty": 0,
        "skipped_invalid_structure": 0,
        "skipped_no_assistant": 0,
        "messages_removed": {
            "system": 0,
            "tool": 0,
            "other": 0
        }
    }
    
    print(f"\nProcessing and filtering to user→assistant pairs...")
    print(f"Output: {OUTPUT_FILE}\n")
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as outfile:
        for entry in tqdm(stream_jsonl(INPUT_FILE), total=total_lines, desc="Processing"):
            stats["total_input"] += 1
            
            messages = entry.get("messages", [])
            
            if not messages:
                stats["skipped_empty"] += 1
                continue
            
            # Count what we're removing
            for msg in messages:
                role = msg.get("role", "")
                if role == "system":
                    stats["messages_removed"]["system"] += 1
                elif role == "tool":
                    stats["messages_removed"]["tool"] += 1
                elif role not in ("user", "assistant"):
                    stats["messages_removed"]["other"] += 1
            
            # Filter to user/assistant only
            filtered_messages = filter_to_user_assistant(messages)
            
            # Validate structure
            if not validate_conversation(filtered_messages):
                if not any(m["role"] == "assistant" for m in filtered_messages):
                    stats["skipped_no_assistant"] += 1
                else:
                    stats["skipped_invalid_structure"] += 1
                continue
            
            # Write filtered conversation
            output_entry = {"messages": filtered_messages}
            outfile.write(json.dumps(output_entry, ensure_ascii=False) + '\n')
            stats["total_output"] += 1
    
    # Save statistics
    with open(STATS_FILE, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE")
    print("="*60)
    print(f"Input entries:  {stats['total_input']:,}")
    print(f"Output entries: {stats['total_output']:,}")
    print(f"Retention rate: {stats['total_output']/stats['total_input']*100:.2f}%")
    print("\nSkipped entries:")
    print(f"  - Empty messages:       {stats['skipped_empty']:,}")
    print(f"  - Invalid structure:    {stats['skipped_invalid_structure']:,}")
    print(f"  - No assistant reply:   {stats['skipped_no_assistant']:,}")
    print("\nMessages removed (per conversation):")
    print(f"  - System messages:      {stats['messages_removed']['system']:,}")
    print(f"  - Tool messages:        {stats['messages_removed']['tool']:,}")
    print(f"  - Other roles:          {stats['messages_removed']['other']:,}")
    print(f"\nStatistics saved to: {STATS_FILE}")
    print(f"Processed data saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    process_dataset()
