
# microsoft/bitnet-b1.58-2B-4T-bf16

"""
Download microsoft/bitnet-b1.58-2B-4T-bf16 weights to ./models/bitnet-b1.58-2b-4t-bf16/

Prerequisites:
  1. Accept Microsoft's license at https://huggingface.co/microsoft/bitnet-b1.58-2B-4T-bf16
  2. Set HF_BITNET_TOKEN env var:  export HF_BITNET_TOKEN=hf_...

Run:
  python bitnet.py
"""

import os
import sys
from dotenv import load_dotenv
from huggingface_hub import snapshot_download

load_dotenv(".env")  # Load environment variables from .env file

MODEL_ID = "microsoft/bitnet-b1.58-2B-4T-bf16"
LOCAL_DIR = "./models/bitnet-b1.58-2b-4t-bf16"

token = os.environ.get("HF_BITNET_TOKEN")
if not token:
    print("Error: HF_BITNET_TOKEN environment variable is not set.")
    print("Get your token from https://huggingface.co/settings/tokens")
    sys.exit(1)

print(f"Downloading {MODEL_ID} to {LOCAL_DIR} ...")
print("This is ~16GB — it may take a while on first run.\n")

snapshot_download(
    repo_id=MODEL_ID,
    local_dir=LOCAL_DIR,
    token=token,
    ignore_patterns=["*.pt", "original/*"],  # skip redundant PyTorch shards
)

print(f"\nDone. Weights saved to {LOCAL_DIR}")