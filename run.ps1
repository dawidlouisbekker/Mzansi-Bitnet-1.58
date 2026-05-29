# ── Dataset creation ──────────────────────────────────────────────────────────
# Step 1: generate SA 11-language dataset and push to Hugging Face
#   python create_dataset.py --username YOUR_HF_USERNAME --token HF_TOKEN
#
# Local-only (no HF upload):
#   python create_dataset.py --skip-upload

python create_dataset.py --username YOUR_HF_USERNAME --token HF_TOKEN

# ── Overnight training ────────────────────────────────────────────────────────
# Step 2: fine-tune BitNet b1.58 2B with LoRA on all data in ./data
#   No Hugging Face login required — reads from locally downloaded model.
#
# Default run (auto-selects CUDA or CPU, 3 epochs):
#   python train.py
#
# Force CPU (safe for any GPU/VRAM size):
#   python train.py --device cpu
#
# Resume a previous run:
#   python train.py --resume ./checkpoints/latest.pt
#
# Longer overnight run:
#   python train.py --epochs 5 --lora-rank 32 --lora-alpha 64

python train.py --device cpu
