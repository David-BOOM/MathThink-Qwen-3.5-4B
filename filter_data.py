import json
import os

input_file = './Nemotron-Math-v3/train.jsonl'
output_file = './Nemotron-Math-v3/train_filtered.jsonl'

def filter_jsonl():
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    removed_count = 0
    kept_count = 0

    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            if not line.strip():
                continue
                
            try:
                data = json.loads(line)
                # Check if tool_usage is "with Python TIR"
                if data.get("tool_usage") == "with Python TIR":
                    removed_count += 1
                else:
                    outfile.write(line)
                    kept_count += 1
            except json.JSONDecodeError:
                print("Warning: Skipping invalid JSON line.")

    # Replace the original file with the filtered one
    os.replace(output_file, input_file)
    print(f"Filtering complete. Kept {kept_count} entries. Removed {removed_count} entries.")

if __name__ == "__main__":
    filter_jsonl()