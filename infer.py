#!/usr/bin/env python3
"""
infer.py — Standalone BitNet b1.58 2B chat inference.

Usage:
  python infer.py                                    # interactive chat loop
  python infer.py --prompt "What is 2+2?"           # single-turn
  python infer.py --system "Reply only in French."  # with system prompt
  python infer.py --model ./models/bitnet-b1.58-2b-4t-bf16 --max-new-tokens 512

Requirements: torch, transformers, safetensors  (all in requirements.txt)
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoTokenizer

# ── Model constants (from config.json) ────────────────────────────────────────

HIDDEN       = 2560
N_LAYERS     = 30
N_HEADS      = 20
N_KV_HEADS   = 5
HEAD_DIM     = HIDDEN // N_HEADS   # 128
GQA_REPEAT   = N_HEADS // N_KV_HEADS  # 4
INTERMEDIATE = 6912
VOCAB        = 128_256
MAX_SEQ      = 4096
ROPE_THETA   = 500_000.0
NORM_EPS     = 1e-5

BOS_ID  = 128_000
EOS_IDS = {128_001, 128_009}   # <|end_of_text|>  and  <|eot_id|>

# ── RMSNorm ────────────────────────────────────────────────────────────────────

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.float().pow(2).mean(-1, keepdim=True).add(NORM_EPS).sqrt()
        return (x.float() / rms * self.weight.float()).to(x.dtype)

# ── BitLinear ─────────────────────────────────────────────────────────────────

class BitLinear(torch.nn.Linear):
    """nn.Linear with online W1.58A8 quantization matching the training regime."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight.float()
        w_scale = w.abs().mean().clamp(min=1e-8)
        w_q = (w / w_scale).round().clamp(-1, 1)

        xf = x.float()
        x_scale = xf.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 127.0
        x_q = (xf / x_scale).round().clamp(-128, 127)

        out = F.linear(x_q, w_q) * x_scale * w_scale
        return out.to(x.dtype)

# ── RoPE ──────────────────────────────────────────────────────────────────────

def _build_rope(device: torch.device):
    half = HEAD_DIM // 2
    freqs = 1.0 / (ROPE_THETA ** (torch.arange(0, half, device=device).float() / half))
    pos   = torch.arange(MAX_SEQ, device=device).float()
    ang   = torch.outer(pos, freqs)          # [MAX_SEQ, half]
    emb   = torch.cat([ang, ang], dim=-1)    # [MAX_SEQ, HEAD_DIM]
    return emb.cos(), emb.sin()

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], dim=-1)

def _apply_rope(q, k, cos, sin, offset: int):
    seq = q.shape[2]
    c = cos[offset: offset + seq].unsqueeze(0).unsqueeze(0)  # [1,1,T,D]
    s = sin[offset: offset + seq].unsqueeze(0).unsqueeze(0)
    return q * c + _rotate_half(q) * s, k * c + _rotate_half(k) * s

# ── Attention ─────────────────────────────────────────────────────────────────

class Attention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        kv_dim = N_KV_HEADS * HEAD_DIM
        self.q_proj        = BitLinear(HIDDEN, HIDDEN,  bias=False)
        self.k_proj        = BitLinear(HIDDEN, kv_dim,  bias=False)
        self.v_proj        = BitLinear(HIDDEN, kv_dim,  bias=False)
        self.o_proj        = BitLinear(HIDDEN, HIDDEN,  bias=False)
        self.attn_sub_norm = RMSNorm(HIDDEN)

    def forward(self, x, cos, sin, offset, k_cache=None, v_cache=None):
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, N_HEADS,    HEAD_DIM).transpose(1, 2)
        k = self.k_proj(x).view(B, T, N_KV_HEADS, HEAD_DIM).transpose(1, 2)
        v = self.v_proj(x).view(B, T, N_KV_HEADS, HEAD_DIM).transpose(1, 2)

        q, k = _apply_rope(q, k, cos, sin, offset)

        if k_cache is not None:
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)

        # GQA: each KV head serves GQA_REPEAT query heads
        kr = k.repeat_interleave(GQA_REPEAT, dim=1)
        vr = v.repeat_interleave(GQA_REPEAT, dim=1)

        scale = HEAD_DIM ** -0.5
        attn  = torch.matmul(q, kr.transpose(-2, -1)) * scale

        if T > 1:
            S = kr.shape[2]
            mask = torch.full((T, S), float("-inf"), device=x.device, dtype=x.dtype)
            mask = torch.triu(mask, diagonal=S - T + 1)
            attn = attn + mask

        attn = F.softmax(attn.float(), dim=-1).to(x.dtype)
        out  = torch.matmul(attn, vr).transpose(1, 2).reshape(B, T, HIDDEN)
        out  = self.attn_sub_norm(out)
        return self.o_proj(out), k, v

# ── MLP ───────────────────────────────────────────────────────────────────────

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

# ── Decoder layer ─────────────────────────────────────────────────────────────

class DecoderLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layernorm          = RMSNorm(HIDDEN)
        self.self_attn                = Attention()
        self.post_attention_layernorm = RMSNorm(HIDDEN)
        self.mlp                      = MLP()

    def forward(self, x, cos, sin, offset, k_cache=None, v_cache=None):
        h, new_k, new_v = self.self_attn(
            self.input_layernorm(x), cos, sin, offset, k_cache, v_cache
        )
        x = x + h
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x, new_k, new_v

# ── Full model ────────────────────────────────────────────────────────────────

class BitNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(VOCAB, HIDDEN)
        self.layers       = torch.nn.ModuleList([DecoderLayer() for _ in range(N_LAYERS)])
        self.norm         = RMSNorm(HIDDEN)

    def forward(self, input_ids, offset=0, k_caches=None, v_caches=None):
        x   = self.embed_tokens(input_ids)
        cos, sin = _build_rope(x.device)

        new_k_caches, new_v_caches = [], []
        for i, layer in enumerate(self.layers):
            kc = k_caches[i] if k_caches else None
            vc = v_caches[i] if v_caches else None
            x, nk, nv = layer(x, cos, sin, offset, kc, vc)
            new_k_caches.append(nk)
            new_v_caches.append(nv)

        x      = self.norm(x)
        logits = F.linear(x, self.embed_tokens.weight)  # tied weights
        return logits, new_k_caches, new_v_caches

# ── Weight loading ────────────────────────────────────────────────────────────

def load_model(model_dir: str, device: torch.device) -> BitNet:
    weights_path = Path(model_dir) / "model.safetensors"
    print(f"Loading weights from {weights_path} …", flush=True)

    raw   = load_file(str(weights_path))
    state = {k.removeprefix("model."): v for k, v in raw.items()}

    model = BitNet()
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"Warning — missing keys: {missing[:5]}", flush=True)
    if unexpected:
        print(f"Warning — unexpected keys: {unexpected[:5]}", flush=True)

    model = model.to(dtype=torch.bfloat16, device=device).eval()
    print(f"Model ready on {device}.", flush=True)
    return model

# ── Sampling ──────────────────────────────────────────────────────────────────

def sample(logits: torch.Tensor, temperature: float, top_p: float) -> int:
    if temperature <= 0.0:
        return int(logits.argmax().item())

    logits = logits.float() / temperature
    probs  = F.softmax(logits, dim=-1)

    # Top-p (nucleus) filtering
    sorted_probs, sorted_idx = probs.sort(descending=True)
    cumsum = sorted_probs.cumsum(dim=-1)
    sorted_probs[cumsum - sorted_probs > top_p] = 0.0
    sorted_probs.div_(sorted_probs.sum())

    chosen = sorted_idx[torch.multinomial(sorted_probs, 1)].item()
    return int(chosen)

# ── Generation ────────────────────────────────────────────────────────────────

@torch.inference_mode()
def generate(
    model: BitNet,
    tokenizer,
    prompt_ids: list[int],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    device: torch.device,
) -> str:
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    logits, k_caches, v_caches = model(ids, offset=0)
    offset = ids.shape[1]

    generated: list[int] = []
    prev_text  = ""

    for _ in range(max_new_tokens):
        next_id = sample(logits[0, -1], temperature, top_p)

        if next_id in EOS_IDS:
            break

        generated.append(next_id)

        # Stream: decode all generated tokens, print the new suffix
        cur_text = tokenizer.decode(generated, skip_special_tokens=True)
        if len(cur_text) > len(prev_text):
            print(cur_text[len(prev_text):], end="", flush=True)
            prev_text = cur_text

        next_tensor = torch.tensor([[next_id]], dtype=torch.long, device=device)
        logits, k_caches, v_caches = model(
            next_tensor, offset=offset, k_caches=k_caches, v_caches=v_caches
        )
        offset += 1

    print()  # newline after streamed output
    return tokenizer.decode(generated, skip_special_tokens=True)

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="BitNet b1.58 2B inference")
    parser.add_argument("--model",          default="./models/bitnet-b1.58-2b-4t-bf16")
    parser.add_argument("--max-new-tokens", type=int,   default=512)
    parser.add_argument("--temperature",    type=float, default=0.6)
    parser.add_argument("--top-p",          type=float, default=0.9)
    parser.add_argument("--system",         default="",
                        help="Optional system prompt prepended to every conversation")
    parser.add_argument("--prompt",         default="",
                        help="Single-turn prompt (skips interactive loop)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model     = load_model(args.model, device)

    def run_turn(history: list[dict]) -> str:
        prompt_text = tokenizer.apply_chat_template(
            history, tokenize=False, add_generation_prompt=True
        )
        prompt_ids = [BOS_ID] + tokenizer.encode(prompt_text, add_special_tokens=False)
        print("\nAssistant: ", end="", flush=True)
        reply = generate(
            model, tokenizer, prompt_ids,
            args.max_new_tokens, args.temperature, args.top_p, device
        )
        return reply

    history: list[dict] = []
    if args.system:
        history.append({"role": "system", "content": args.system})

    if args.prompt:
        history.append({"role": "user", "content": args.prompt})
        reply = run_turn(history)
        history.append({"role": "assistant", "content": reply})
        return

    # Interactive chat loop
    print("BitNet b1.58 2B — type 'exit' or Ctrl-C to quit.\n")
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_input or user_input.lower() in {"exit", "quit"}:
            break
        history.append({"role": "user", "content": user_input})
        reply = run_turn(history)
        history.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
