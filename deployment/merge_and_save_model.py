"""
Merge LoRA adapter with base model and save locally.
This creates a fully merged model that doesn't need HuggingFace downloads.

Run this once to create a completely local model:
    python deployment/merge_and_save_model.py

Then update api.py to use the merged model.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

print("="*80)
print("MERGE LORA ADAPTER WITH BASE MODEL")
print("="*80)

# Paths
base_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
lora_adapter_path = project_root / "notebooks" / "medical_llm_final"
output_dir = project_root / "models" / "merged_model"

print(f"\nBase model: {base_model_name}")
print(f"LoRA adapter: {lora_adapter_path}")
print(f"Output directory: {output_dir}")
print("\n" + "="*80)

# Create output directory
output_dir.mkdir(parents=True, exist_ok=True)

# Step 1: Load base model
print("\n[1/4] Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16,
    device_map="cpu",  # Load to CPU for merging
    trust_remote_code=True
)
print("✓ Base model loaded")

# Step 2: Load tokenizer
print("\n[2/4] Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print("✓ Tokenizer loaded")

# Step 3: Load and merge LoRA adapter
print("\n[3/4] Loading LoRA adapter...")
model = PeftModel.from_pretrained(model, str(lora_adapter_path))
print("✓ LoRA adapter loaded")

print("\n[3/4] Merging LoRA weights with base model...")
model = model.merge_and_unload()
print("✓ Weights merged successfully")

# Step 4: Save merged model
print(f"\n[4/4] Saving merged model to {output_dir}...")
model.save_pretrained(str(output_dir))
tokenizer.save_pretrained(str(output_dir))
print("✓ Model saved")

print("\n" + "="*80)
print("SUCCESS! Merged model saved locally.")
print("="*80)
print("\nTo use the merged model, update deployment/api.py:")
print('    "base_model_name": "models/merged_model",')
print('    "lora_adapter_path": None,  # No adapter needed')
print("\nThis will eliminate all HuggingFace downloads!")
print("="*80)
