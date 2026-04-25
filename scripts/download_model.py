"""
Model Download Script for Qwen 3.5 4B
Downloads the base model from HuggingFace for fine-tuning.
"""

import os
import sys
from pathlib import Path

# Try importing required libraries
try:
    from huggingface_hub import snapshot_download, login
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
except ImportError as e:
    print(f"Error: Missing required library: {e}")
    print("Please install: pip install huggingface_hub transformers")
    sys.exit(1)


MODEL_NAME = "Qwen/Qwen3.5-4B"
MODEL_DIR = "./outputs/models/base_model"
CACHE_DIR = "./outputs/models/cache"


def check_auth():
    """Check if HuggingFace authentication is needed."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        print("Using HuggingFace token from environment variable.")
        return token
    
    # Check if already logged in
    try:
        from huggingface_hub import HfFolder
        token = HfFolder.get_token()
        if token:
            print("Using cached HuggingFace token.")
            return token
    except:
        pass
    
    print("No HuggingFace token found.")
    print("If the model requires authentication, run: huggingface-cli login")
    return None


def download_model():
    """Download the Qwen 3.5 4B model."""
    
    print("="*60)
    print(f"Downloading Model: {MODEL_NAME}")
    print("="*60)
    
    # Create directories
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    # Check authentication
    token = check_auth()
    
    print(f"\nModel: {MODEL_NAME}")
    print(f"Destination: {MODEL_DIR}")
    print(f"Cache: {CACHE_DIR}")
    
    try:
        # First, verify model exists and get config
        print("\n[1/3] Fetching model configuration...")
        config = AutoConfig.from_pretrained(
            MODEL_NAME,
            cache_dir=CACHE_DIR,
            token=token,
            trust_remote_code=True
        )
        print(f"  Model type: {config.model_type}")
        # Handle different config attribute names
        hidden_size = getattr(config, 'hidden_size', None) or getattr(config, 'd_model', None) or getattr(config, 'dim', 'N/A')
        num_layers = getattr(config, 'num_hidden_layers', None) or getattr(config, 'n_layer', 'N/A')
        vocab_size = getattr(config, 'vocab_size', 'N/A')
        print(f"  Hidden size: {hidden_size}")
        print(f"  Num layers: {num_layers}")
        print(f"  Vocab size: {vocab_size}")
        
        # Download tokenizer
        print("\n[2/3] Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            cache_dir=CACHE_DIR,
            token=token,
            trust_remote_code=True
        )
        tokenizer.save_pretrained(MODEL_DIR)
        print(f"  Tokenizer saved to: {MODEL_DIR}")
        print(f"  Vocab size: {len(tokenizer)}")
        
        # Download full model snapshot
        print("\n[3/3] Downloading model weights...")
        print("  This may take a while depending on your connection...")
        
        snapshot_path = snapshot_download(
            repo_id=MODEL_NAME,
            local_dir=MODEL_DIR,
            cache_dir=CACHE_DIR,
            token=token,
            ignore_patterns=["*.md", "*.txt", ".gitattributes"],
            resume_download=True  # Resume if interrupted
        )
        
        print(f"\n✓ Model downloaded successfully!")
        print(f"  Location: {snapshot_path}")
        
        # List downloaded files
        print("\nDownloaded files:")
        model_path = Path(MODEL_DIR)
        for f in sorted(model_path.glob("*")):
            if f.is_file():
                size_mb = f.stat().st_size / (1024 * 1024)
                print(f"  - {f.name}: {size_mb:.1f} MB")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error downloading model: {e}")
        print("\nTroubleshooting:")
        print("1. Check your internet connection")
        print(f"2. Verify the model name is correct: {MODEL_NAME}")
        print("3. If model requires authentication: huggingface-cli login")
        print("4. Check HuggingFace status: https://status.huggingface.co/")
        return False


def main():
    """Main entry point."""
    success = download_model()
    
    if success:
        print("\n" + "="*60)
        print("DOWNLOAD COMPLETE")
        print("="*60)
        print(f"Model ready at: {MODEL_DIR}")
        print("You can now run the training script.")
    else:
        print("\n" + "="*60)
        print("DOWNLOAD FAILED")
        print("="*60)
        sys.exit(1)


if __name__ == "__main__":
    main()
