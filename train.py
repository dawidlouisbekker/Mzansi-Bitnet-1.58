#!/usr/bin/env python3
"""
train.py — Overnight LoRA fine-tuning of BitNet b1.58 2B.

No Hugging Face login required — reads weights from a locally downloaded model directory.
Loads all *.jsonl files found in --data-dir, attaches LoRA adapters to the attention
layers, trains, and saves checkpoints to --output-dir.

Usage:
  python train.py                            # all defaults, auto-selects CUDA/CPU
  python train.py --epochs 5
  python train.py --device cpu               # force CPU (safe for any VRAM size)
  python train.py --resume ./checkpoints/latest.pt
  python train.py --lora-rank 32 --lora-alpha 64

After training, load the adapter with load_lora_checkpoint() and merge if needed.
"""

import argparse
import json
import logging
import math
import signal
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoTokenizer

# ── Model constants (must match config.json) ─────────────────────────────────

HIDDEN       = 2560
N_LAYERS     = 30
N_HEADS      = 20
N_KV_HEADS   = 5
HEAD_DIM     = HIDDEN // N_HEADS        # 128
GQA_REPEAT   = N_HEADS // N_KV_HEADS   # 4
INTERMEDIATE = 6912
VOCAB        = 128_256
MAX_SEQ      = 4096
ROPE_THETA   = 500_000.0
NORM_EPS     = 1e-5

# ── Model architecture (identical to infer.py) ────────────────────────────────

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.float().pow(2).mean(-1, keepdim=True).add(NORM_EPS).sqrt()
        return (x.float() / rms * self.weight.float()).to(x.dtype)


class BitLinear(torch.nn.Linear):
    """nn.Linear with online W1.58A8 quantization (straight-through on backward)."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight.float()
        w_scale = w.abs().mean().clamp(min=1e-8)
        w_q = (w / w_scale).round().clamp(-1, 1)

        xf = x.float()
        x_scale = xf.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 127.0
        x_q = (xf / x_scale).round().clamp(-128, 127)

        out = F.linear(x_q, w_q) * x_scale * w_scale
        return out.to(x.dtype)


def _build_rope(device: torch.device):
    half = HEAD_DIM // 2
    freqs = 1.0 / (ROPE_THETA ** (torch.arange(0, half, device=device).float() / half))
    pos   = torch.arange(MAX_SEQ, device=device).float()
    ang   = torch.outer(pos, freqs)
    emb   = torch.cat([ang, ang], dim=-1)
    return emb.cos(), emb.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], dim=-1)


def _apply_rope(q, k, cos, sin, offset: int):
    seq = q.shape[2]
    c = cos[offset: offset + seq].unsqueeze(0).unsqueeze(0)
    s = sin[offset: offset + seq].unsqueeze(0).unsqueeze(0)
    return q * c + _rotate_half(q) * s, k * c + _rotate_half(k) * s


class Attention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        kv_dim = N_KV_HEADS * HEAD_DIM
        self.q_proj        = BitLinear(HIDDEN, HIDDEN,  bias=False)
        self.k_proj        = BitLinear(HIDDEN, kv_dim,  bias=False)
        self.v_proj        = BitLinear(HIDDEN, kv_dim,  bias=False)
        self.o_proj        = BitLinear(HIDDEN, HIDDEN,  bias=False)
        self.attn_sub_norm = RMSNorm(HIDDEN)

    def forward(self, x, cos, sin, offset=0):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, N_HEADS,    HEAD_DIM).transpose(1, 2)
        k = self.k_proj(x).view(B, T, N_KV_HEADS, HEAD_DIM).transpose(1, 2)
        v = self.v_proj(x).view(B, T, N_KV_HEADS, HEAD_DIM).transpose(1, 2)

        q, k = _apply_rope(q, k, cos, sin, offset)

        kr = k.repeat_interleave(GQA_REPEAT, dim=1)
        vr = v.repeat_interleave(GQA_REPEAT, dim=1)

        scale = HEAD_DIM ** -0.5
        attn  = torch.matmul(q, kr.transpose(-2, -1)) * scale

        if T > 1:
            mask = torch.full((T, T), float("-inf"), device=x.device, dtype=x.dtype)
            mask = torch.triu(mask, diagonal=1)
            attn = attn + mask

        attn = F.softmax(attn.float(), dim=-1).to(x.dtype)
        out  = torch.matmul(attn, vr).transpose(1, 2).reshape(B, T, HIDDEN)
        out  = self.attn_sub_norm(out)
        return self.o_proj(out)


class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj    = BitLinear(HIDDEN,       INTERMEDIATE, bias=False)
        self.up_proj      = BitLinear(HIDDEN,       INTERMEDIATE, bias=False)
        self.down_proj    = BitLinear(INTERMEDIATE, HIDDEN,       bias=False)
        self.ffn_sub_norm = RMSNorm(INTERMEDIATE)

    def forward(self, x):
        h = F.relu(self.gate_proj(x)).pow(2) * self.up_proj(x)
        h = self.ffn_sub_norm(h)
        return self.down_proj(h)


class DecoderLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layernorm          = RMSNorm(HIDDEN)
        self.self_attn                = Attention()
        self.post_attention_layernorm = RMSNorm(HIDDEN)
        self.mlp                      = MLP()

    def forward(self, x, cos, sin, offset=0):
        h   = self.self_attn(self.input_layernorm(x), cos, sin, offset)
        x   = x + h
        x   = x + self.mlp(self.post_attention_layernorm(x))
        return x


class BitNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(VOCAB, HIDDEN)
        self.layers       = torch.nn.ModuleList([DecoderLayer() for _ in range(N_LAYERS)])
        self.norm         = RMSNorm(HIDDEN)

    def forward(self, input_ids: torch.Tensor, offset: int = 0) -> torch.Tensor:
        x        = self.embed_tokens(input_ids)
        cos, sin = _build_rope(x.device)
        for layer in self.layers:
            x = layer(x, cos, sin, offset)
        x      = self.norm(x)
        logits = F.linear(x, self.embed_tokens.weight)  # tied weights
        return logits


# ── Weight loading ────────────────────────────────────────────────────────────

def load_model(model_dir: str, device: torch.device) -> BitNet:
    model_path = Path(model_dir)

    # Support single-file and sharded safetensors
    single = model_path / "model.safetensors"
    shards = sorted(model_path.glob("model-*-of-*.safetensors"))

    if single.exists():
        files = [single]
    elif shards:
        files = shards
    else:
        sys.exit(f"No model.safetensors found in {model_dir}")

    logging.info(f"Loading weights from {model_dir} ({len(files)} shard(s)) ...")
    raw: dict = {}
    for f in files:
        raw.update(load_file(str(f)))

    state = {k.removeprefix("model."): v for k, v in raw.items()}

    model = BitNet()
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        logging.warning(f"Missing keys (first 5): {missing[:5]}")
    if unexpected:
        logging.warning(f"Unexpected keys (first 5): {unexpected[:5]}")

    model = model.to(dtype=torch.bfloat16, device=device)
    model.train()

    # Freeze all base model parameters
    for p in model.parameters():
        p.requires_grad_(False)

    logging.info(f"Base model loaded and frozen on {device}.")
    return model


# ── LoRA ──────────────────────────────────────────────────────────────────────

class LoRALinear(torch.nn.Module):
    """Parallel LoRA branch added to a frozen BitLinear layer."""

    def __init__(self, base: torch.nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base  = base
        self.scale = alpha / rank
        d_in  = base.weight.shape[1]
        d_out = base.weight.shape[0]
        # A initialised with small Gaussian; B initialised to zero (LoRA paper §4)
        self.lora_A = torch.nn.Parameter(torch.randn(rank, d_in) * 0.02)
        self.lora_B = torch.nn.Parameter(torch.zeros(d_out, rank))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + (x @ self.lora_A.t() @ self.lora_B.t()) * self.scale


_LORA_ATTN_ATTRS = ("q_proj", "k_proj", "v_proj", "o_proj")


def inject_lora(model: BitNet, rank: int, alpha: float, device: torch.device) -> None:
    """Replace BitLinear attention projections with LoRALinear in-place."""
    for layer in model.layers:
        attn = layer.self_attn
        for attr in _LORA_ATTN_ATTRS:
            base = getattr(attn, attr)
            lora = LoRALinear(base, rank, alpha).to(dtype=torch.float32, device=device)
            setattr(attn, attr, lora)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"LoRA injected — trainable parameters: {n_params:,} "
                 f"({n_params / 1e6:.2f}M) across {N_LAYERS * len(_LORA_ATTN_ATTRS)} adapters.")


# ── Dataset ───────────────────────────────────────────────────────────────────

def load_all_jsonl(data_dir: str) -> list[list[dict]]:
    """Read all *.jsonl files in data_dir, return list of message lists."""
    files = sorted(Path(data_dir).glob("*.jsonl"))
    if not files:
        sys.exit(f"No .jsonl files found in {data_dir}")

    all_examples: list[list[dict]] = []
    for path in files:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                msgs = obj.get("messages")
                if msgs and isinstance(msgs, list) and len(msgs) >= 2:
                    all_examples.append(msgs)

    logging.info(f"Loaded {len(all_examples)} examples from {len(files)} file(s) in {data_dir}")
    return all_examples


def tokenize_with_labels(
    tokenizer,
    messages: list[dict],
    max_len: int,
) -> tuple[list[int], list[int]]:
    """
    Tokenize a full conversation and build labels.
    Non-assistant tokens are masked with -100 so loss is only computed on
    assistant responses — the same masking strategy as dataset.rs.
    """
    full_ids: list[int] = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False
    )
    full_ids = full_ids[:max_len]
    labels   = [-100] * len(full_ids)

    seen: list[dict] = []
    for msg in messages:
        if msg["role"] == "assistant":
            # Prefix up to this assistant turn (with generation header) marks the
            # exact position where the model must start predicting.
            prefix_ids: list[int] = tokenizer.apply_chat_template(
                seen, tokenize=True, add_generation_prompt=True
            )
            prefix_len = len(prefix_ids)

            # Full sequence up to end of this assistant message
            end_ids: list[int] = tokenizer.apply_chat_template(
                seen + [msg], tokenize=True, add_generation_prompt=False
            )
            end_len = len(end_ids)

            for j in range(prefix_len, min(end_len, max_len)):
                labels[j] = full_ids[j]

        seen.append(msg)

    return full_ids, labels


def build_dataset(
    tokenizer,
    examples: list[list[dict]],
    max_len: int,
) -> list[tuple[list[int], list[int]]]:
    """Tokenize every example; drop any that have no trainable tokens."""
    dataset: list[tuple[list[int], list[int]]] = []
    skipped = 0
    for msgs in examples:
        ids, labels = tokenize_with_labels(tokenizer, msgs, max_len)
        if all(l == -100 for l in labels):
            skipped += 1
            continue
        dataset.append((ids, labels))
    if skipped:
        logging.warning(f"Dropped {skipped} examples with no assistant tokens.")
    logging.info(f"Dataset ready: {len(dataset)} tokenized examples.")
    return dataset


# ── LR schedule ───────────────────────────────────────────────────────────────

def cosine_lr(step: int, total_steps: int, warmup_steps: int, lr: float) -> float:
    if step < warmup_steps:
        return lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return lr * 0.5 * (1.0 + math.cos(math.pi * progress))


# ── Checkpointing ─────────────────────────────────────────────────────────────

def save_checkpoint(
    model: BitNet,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    loss: float,
    output_dir: Path,
    tag: str = "",
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    lora_state = {
        name: param.data.cpu().clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }
    checkpoint = {
        "epoch":           epoch,
        "global_step":     global_step,
        "loss":            loss,
        "lora_state":      lora_state,
        "optimizer_state": optimizer.state_dict(),
    }
    suffix   = f"_{tag}" if tag else ""
    filename = f"checkpoint_epoch{epoch}_step{global_step}{suffix}.pt"
    path     = output_dir / filename
    torch.save(checkpoint, path)

    # Always keep a "latest" pointer for easy resume
    latest = output_dir / "latest.pt"
    torch.save(checkpoint, latest)
    return path


def load_checkpoint(
    path: str,
    model: BitNet,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[int, int, float]:
    """Returns (start_epoch, global_step, last_loss)."""
    ckpt = torch.load(path, map_location=device)

    lora_state = ckpt["lora_state"]
    current_state = dict(model.named_parameters())
    loaded, missing = 0, 0
    for name, tensor in lora_state.items():
        if name in current_state and current_state[name].requires_grad:
            current_state[name].data.copy_(tensor.to(device))
            loaded += 1
        else:
            missing += 1

    optimizer.load_state_dict(ckpt["optimizer_state"])
    epoch       = ckpt["epoch"]
    global_step = ckpt["global_step"]
    loss        = ckpt.get("loss", float("nan"))
    logging.info(f"Resumed from {path} (epoch {epoch}, step {global_step}, "
                 f"loaded {loaded} LoRA tensors, {missing} not found)")
    return epoch, global_step, loss


# ── Training loop ─────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Logging: console + file
    log_path = output_dir / "training_log.txt"
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s  %(levelname)s  %(message)s"))
    logging.getLogger().addHandler(file_handler)
    logging.info(f"Training log: {log_path}")

    device = torch.device(args.device)
    logging.info(f"Device: {device}")

    # Load tokenizer (no HF login — reads from local model directory)
    logging.info(f"Loading tokenizer from {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load + freeze base model, inject LoRA
    model = load_model(args.model, device)
    inject_lora(model, rank=args.lora_rank, alpha=args.lora_alpha, device=device)

    # Only LoRA parameters get an optimizer
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)

    # Data
    raw_examples = load_all_jsonl(args.data_dir)
    dataset      = build_dataset(tokenizer, raw_examples, args.max_seq_len)
    if not dataset:
        sys.exit("No valid training examples after tokenization.")

    total_steps   = (len(dataset) * args.epochs) // args.grad_accum
    warmup_steps  = max(1, int(total_steps * 0.05))
    logging.info(
        f"Training: {args.epochs} epochs, {len(dataset)} examples/epoch, "
        f"{total_steps} optimizer steps, {warmup_steps} warmup steps."
    )

    # Resume if requested
    start_epoch  = 0
    global_step  = 0
    if args.resume:
        start_epoch, global_step, _ = load_checkpoint(
            args.resume, model, optimizer, device
        )

    # Graceful Ctrl-C: save checkpoint before exiting
    interrupted = False
    def _sigint_handler(sig, frame):
        nonlocal interrupted
        interrupted = True
        logging.warning("Interrupt received — will save checkpoint and exit after current step.")
    signal.signal(signal.SIGINT, _sigint_handler)

    # Training
    step_start = time.time()
    accum_loss = 0.0
    accum_tok  = 0

    for epoch in range(start_epoch, args.epochs):
        logging.info(f"─── Epoch {epoch + 1}/{args.epochs} ───")

        for ex_idx, (input_ids, label_ids) in enumerate(dataset):
            if interrupted:
                break

            ids_t    = torch.tensor([input_ids], dtype=torch.long,  device=device)
            labels_t = torch.tensor([label_ids], dtype=torch.long,  device=device)

            logits = model(ids_t)                              # [1, T, VOCAB]
            shift_logits = logits[0, :-1, :].float()          # [T-1, VOCAB]
            shift_labels = labels_t[0, 1:]                    # [T-1]

            loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)
            loss_scaled = loss / args.grad_accum
            loss_scaled.backward()

            n_real = (shift_labels != -100).sum().item()
            accum_loss += loss.item() * n_real
            accum_tok  += n_real

            # Optimizer step every grad_accum examples
            if (ex_idx + 1) % args.grad_accum == 0 or ex_idx == len(dataset) - 1:
                # LR update
                lr_now = cosine_lr(global_step, total_steps, warmup_steps, args.lr)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr_now

                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                avg_loss = accum_loss / max(accum_tok, 1)
                accum_loss = 0.0
                accum_tok  = 0

                elapsed = time.time() - step_start
                steps_done  = global_step
                steps_left  = total_steps - steps_done
                eta_sec     = (elapsed / max(steps_done, 1)) * steps_left
                eta_str     = _fmt_duration(eta_sec)

                msg = (
                    f"epoch {epoch + 1}/{args.epochs}  "
                    f"step {global_step}/{total_steps}  "
                    f"loss {avg_loss:.4f}  "
                    f"lr {lr_now:.2e}  "
                    f"ETA {eta_str}"
                )
                logging.info(msg)

                # Periodic checkpoint
                if global_step % args.save_steps == 0:
                    p = save_checkpoint(model, optimizer, epoch, global_step,
                                        avg_loss, output_dir)
                    logging.info(f"  Checkpoint saved → {p.name}")

        if not interrupted:
            # End-of-epoch checkpoint
            p = save_checkpoint(model, optimizer, epoch + 1, global_step,
                                accum_loss / max(accum_tok, 1) if accum_tok else 0.0,
                                output_dir, tag="epoch_end")
            logging.info(f"Epoch {epoch + 1} complete. Checkpoint → {p.name}")
        else:
            p = save_checkpoint(model, optimizer, epoch, global_step, 0.0,
                                output_dir, tag="interrupted")
            logging.info(f"Interrupted. Checkpoint saved → {p.name}")
            break

    total_time = _fmt_duration(time.time() - step_start)
    logging.info(f"Training finished in {total_time}. Checkpoints in {output_dir.resolve()}")


def _fmt_duration(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s   = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Overnight LoRA fine-tuning of BitNet b1.58 2B on SA language data."
    )
    p.add_argument("--model",      default="./models/bitnet-b1.58-2b-4t-bf16",
                   help="Path to locally downloaded model directory (default: ./models/bitnet-b1.58-2b-4t-bf16)")
    p.add_argument("--data-dir",   default="./data",
                   help="Directory to scan for *.jsonl training files (default: ./data)")
    p.add_argument("--output-dir", default="./checkpoints",
                   help="Directory for checkpoints and training log (default: ./checkpoints)")
    p.add_argument("--resume",     default=None,
                   help="Path to a checkpoint .pt file to resume training from")
    p.add_argument("--epochs",     type=int,   default=3,
                   help="Number of full passes over the dataset (default: 3)")
    p.add_argument("--lora-rank",  type=int,   default=16,
                   help="LoRA rank r (default: 16)")
    p.add_argument("--lora-alpha", type=float, default=32.0,
                   help="LoRA alpha (default: 32.0 — scaling = alpha/rank = 2.0)")
    p.add_argument("--lr",         type=float, default=2e-4,
                   help="Peak learning rate for AdamW (default: 2e-4)")
    p.add_argument("--grad-accum", type=int,   default=8,
                   help="Gradient accumulation steps (default: 8)")
    p.add_argument("--max-seq-len",type=int,   default=512,
                   help="Maximum token sequence length per example (default: 512)")
    p.add_argument("--save-steps", type=int,   default=100,
                   help="Save a checkpoint every N optimizer steps (default: 100)")
    p.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu",
                   help="Device: 'cuda' or 'cpu' (auto-detected by default)")
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    train(parse_args())
