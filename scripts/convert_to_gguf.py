#!/usr/bin/env python3
"""
Convert a merged Hugging Face model to GGUF format using raw llama.cpp.
This completely bypasses Unsloth.
"""

import os
import sys
import json
import shutil
import zipfile
import argparse
import subprocess
import urllib.request
from pathlib import Path

def project_root() -> Path:
    return Path(__file__).resolve().parent.parent

def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (project_root() / path).resolve()

def setup_llama_cpp(llama_cpp_dir: Path):
    """Clone llama.cpp and download the Windows CPU precompiled binaries."""
    print("Checking local llama.cpp installation...")
    llama_cpp_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Clone repo if needed
    convert_script = llama_cpp_dir / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        print(f"Cloning llama.cpp to {llama_cpp_dir}...")
        subprocess.run(["git", "clone", "https://github.com/ggml-org/llama.cpp.git", str(llama_cpp_dir)], check=True)
    
    # 2. Install python dependencies
    print("Ensuring llama.cpp Python dependencies are installed...")
    req_file = llama_cpp_dir / "requirements.txt"
    if req_file.exists():
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(req_file)], check=True)

    # 3. Download the llama-quantize binary if needed for quantization
    quantize_exe = llama_cpp_dir / "llama-quantize.exe"
    if not quantize_exe.exists():
        print("Fetching latest Windows binaries for llama-quantize (CPU version for broad compatibility)...")
        req = urllib.request.Request("https://github.com/ggml-org/llama.cpp/releases/latest", method="HEAD")
        try:
            resp = urllib.request.urlopen(req)
            url = resp.geturl()
        except urllib.error.HTTPError as e:
            # Handle redirect manually just in case
            url = e.url if hasattr(e, "url") else e.headers.get("Location", "https://github.com/ggml-org/llama.cpp/releases/tag/b8763")
        except Exception:
            url = "https://github.com/ggml-org/llama.cpp/releases/tag/b8763"
            
        tag = url.split("/")[-1]
        download_url = f"https://github.com/ggml-org/llama.cpp/releases/download/{tag}/llama-{tag}-bin-win-cpu-x64.zip"
        zip_path = llama_cpp_dir / "llama-bin.zip"
        
        print(f"Downloading {download_url}...")
        urllib.request.urlretrieve(download_url, str(zip_path))
        
        print("Extracting llama-quantize.exe and required DLLs...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            # Look for the executable and any DLLs inside the zip
            for member in zip_ref.namelist():
                if member.endswith("llama-quantize.exe") or member.endswith(".dll"):
                    zip_ref.extract(member, str(llama_cpp_dir))
                    extracted_path = llama_cpp_dir / member
                    
                    # Move to root of llama_cpp_dir if it extracted into a subdirectory
                    target_path = llama_cpp_dir / Path(member).name
                    if extracted_path != target_path:
                        shutil.move(str(extracted_path), str(target_path))
        
        # Clean up zip and any leftover subdirectories from zip extraction
        zip_path.unlink()
        for item in llama_cpp_dir.iterdir():
            if item.is_dir() and item.name.startswith("llama-"):
                shutil.rmtree(item)

def main():
    parser = argparse.ArgumentParser(description="Convert Hugging Face model to GGUF using llama.cpp natively")
    parser.add_argument(
        "--model-path", 
        type=str, 
        default="./outputs/models/final_model_merged",
        help="Path to the merged Hugging Face model"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="./outputs/final_model",
        help="Directory to save the resulting GGUF file"
    )
    parser.add_argument(
        "--quantization", 
        type=str, 
        default="q4_k_m", 
        help="Quantization precision (e.g., q4_k_m, q8_0, f16, q5_k_m)"
    )
    
    args = parser.parse_args()

    model_path = resolve_path(args.model_path)
    output_dir = resolve_path(args.output_dir)
    llama_cpp_dir = project_root() / ".llama.cpp"
    
    print(f"Input model path: {model_path}")
    print(f"Output directory: {output_dir}")
    print(f"Target quantization: {args.quantization}\n")

    if not model_path.exists():
        print(f"Error: Model path not found at {model_path}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepares llama.cpp tooling automatically
    setup_llama_cpp(llama_cpp_dir)
    
    final_output_path = output_dir / f"{model_path.name}-{args.quantization.upper()}.gguf"
    
    if args.quantization.lower() in ("f16", "f32", "bf16", "q8_0"):
        # F16, F32, BF16 and Q8_0 can be generated directly by the python script
        print(f"\nRunning direct conversion to {args.quantization.upper()}...")
        convert_script = llama_cpp_dir / "convert_hf_to_gguf.py"
        subprocess.run([
            sys.executable, str(convert_script),
            str(model_path),
            "--outfile", str(final_output_path),
            "--outtype", args.quantization.lower()
        ], check=True)
        print(f"\n✅ Conversion completed: {final_output_path}")
        
    else:
        # Complex quantizations (q4_k_m, q5_k_m, etc.) require 2 steps:
        # 1. Convert Hugging Face structure to F16 GGUF
        temp_path = output_dir / "temp-f16.gguf"
        print("\nStep 1: Translating Hugging Face tensors to unquantized F16 GGUF...")
        convert_script = llama_cpp_dir / "convert_hf_to_gguf.py"
        subprocess.run([
            sys.executable, str(convert_script),
            str(model_path),
            "--outfile", str(temp_path),
            "--outtype", "f16"
        ], check=True)
        
        # 2. Use the compiled C++ executable to aggressively quantize the weights
        print(f"\nStep 2: Quantizing the temporary F16 GGUF into {args.quantization.upper()}...")
        quantize_exe = llama_cpp_dir / "llama-quantize.exe"
        subprocess.run([
            str(quantize_exe),
            str(temp_path),
            str(final_output_path),
            args.quantization.upper()
        ], check=True)
        
        # 3. Clean up the large F16 model
        print("\nCleaning up temporary F16 footprint...")
        if temp_path.exists():
            temp_path.unlink()
            
        print(f"\n✅ Build sequence completed for: {final_output_path}")

if __name__ == "__main__":
    main()
