# Qwen 3.5 4B Fine-Tuning reproduction process

This repository stores programs required for reproducing [DavidOKB/MathThink-Qwen-3.5-4B](https://doi.org/10.57967/hf/8542).

Reproducible pipeline for:

1. dataset preprocessing
2. QLoRA training with `scripts\train_ram.py`
3. merging adapter and converting to GGUF format
4. benchmark evaluation via `evaluation\run_all_evals_lmstudio.py` (LM Studio is required for these evaluation programs)

This README provides a thorough step-by-step instruction to reproduce the fine-tuning and evaluation workflow. All inference jobs are configured to run with LM Studio as the backend.

---

## 1) Scope and source of truth

- Base model: `Qwen/Qwen3.5-4B`
- Dataset workflow target: `nvidia/Nemotron-SFT-Math-v3` (***Renamed as `Nemotron-Math-v3`for easy working in this repository***)
- Training entrypoint: `scripts\train_ram.py`
- Evaluation entrypoints: `evaluation\run_all_evals_lmstudio.py`

---

## 2) Prerequisites

- Windows + Python 3.10+
- CUDA GPU recommended for training
- **LM Studio** installed and configured for serving the local model
- Run all commands from repository root (`F:\AI\4016Project`)

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 3) Download Base Model

First, you need to download and cache the Qwen 3.5 4B base model locally.

```bash
python scripts\download_model.py
```

This writes local model files to `outputs\models\base_model`.
*(If needed, authenticate with `huggingface-cli login` first).*

---

## 4) Dataset Preparation

The training requires a filtered, clean dataset.

To reproduce the data preperation steps, you need to download the dataset `nvidia/Nemotron-SFT-Math-v3`manually, and rename it to `Nemotron-Math-v3`. No download program is provided due to heavy RAM consumption (>64GB) during download via git.

1. **Filter tool usage:**
   Remove entries that require tool usage (e.g. "with Python TIR") to ensure the model focuses purely on reasoning.

   ```bash
   python filter_data.py
   ```

   This reads `Nemotron-Math-v3\train.jsonl` and outputs `Nemotron-Math-v3\train_filtered.jsonl`.
2. **Preprocess Dataset:**
   Extract only the user & assistant chat messages, skipping system prompts.

   ```bash
   python scripts\preprocess_dataset.py
   ```

   Note: Ensure `INPUT_FILE` inside `preprocess_dataset.py` points to the filtered dataset (`Nemotron-Math-v3/train_filtered.jsonl`). It will produce `Nemotron-Math-v3\train_processed.jsonl`.

---

## 5) Training

Run the primary training script. This script executes a QLoRA fine-tune targeted at keeping VRAM usage reasonable.

```bash
python scripts\train_ram.py
```

**What happens:**

- Uses the cached base model at `outputs\models\base_model`
- Trains on `Nemotron-Math-v3\train_processed.jsonl`
- Outputs checkpoints and the final adapter to `outputs\models\final_adapter`

---

## 6) Merge and Convert to GGUF

To evaluate the model in LM Studio, you must merge the LoRA adapters into the base model, and then compile it into a `.gguf` file.

1. **Merge the adapter:**

   ```bash
   python scripts\merge_adapter.py
   ```

   This creates a standalone Hugging Face model out of your fine-tune in `outputs\models\final_model_merged`.
2. **Convert to GGUF:**

   ```bash
   python scripts\convert_to_gguf.py
   ```

   This script converts the merged model into GGUF format and saves it. Specifically, look inside `outputs\final_model\` for files like `final_model_merged_alpha*.gguf`.

---

## 7) Start LM Studio Server

1. Open **LM Studio**.
2. Load the specific `.gguf` file you just created (from `outputs\final_model\`). You can also download from LM Studio by searching `DavidOKB/MathThink-Qwen-3.5-4B`
3. Start the Local Inference Server in LM Studio (usually running on `http://localhost:1234`).

---

## 8) Evaluation

Once LM Studio is actively serving your GGUF model, run the evaluation script to test its performance on the benchmarks (GSM8K, MATH, AIME).

```bash
python evaluation\run_all_evals_lmstudio.py
```

### Advanced Evaluation Flags

You can customize the evaluation run using the following flags:

* **Select Specific Benchmarks:**
  Target one or more specific benchmarks instead of running all of them.

  ```bash
  python evaluation\run_all_evals_lmstudio.py --benchmarks gsm8k math
  ```
* **Download Datasets:**
  Download the dataset files if they aren't already locally cached.

  ```bash
  python evaluation\run_all_evals_lmstudio.py --download-datasets
  ```
  *Tip: Use `--download-only` to skip evaluation testing and only grab the datasets.*
* **Limit Max Samples:**
  Restrict the benchmark runs to a certain number of questions (useful for quick testing).

  ```bash
  python evaluation\run_all_evals_lmstudio.py --max-samples 50
  ```
* **Toggle Live Token Progress:**
  Watch the token generation count update in real-time while testing.

  ```bash
  python evaluation\run_all_evals_lmstudio.py --realtime-token-progress
  ```
* **Override Generation Settings:**
  Change the limits for output generation.

  ```bash
  python evaluation\run_all_evals_lmstudio.py --max-output-tokens 1024 --temperature 0.8
