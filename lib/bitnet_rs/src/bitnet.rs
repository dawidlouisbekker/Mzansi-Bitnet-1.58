/// BitNet b1.58 2B architecture.
///
/// Config matches ./models/bitnet-b1.58-2b-4t-bf16/config.json:
///   hidden_size=2560, num_hidden_layers=30, num_attention_heads=20,
///   num_key_value_heads=5, intermediate_size=6912, rope_theta=500000.
///
/// Matrix multiplications in BitLinear layers are dispatched to cuBLAS via
/// cuda_rs when the `cuda` feature is enabled; everything else (norms, RoPE,
/// softmax, residuals) stays on the CPU through candle.
use std::collections::HashMap;

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor, D};

use crate::lora::LoraLinear;

// ---------------------------------------------------------------------------
// Global cuBLAS handle (initialised once, reused for every matmul)
// ---------------------------------------------------------------------------

#[cfg(feature = "cuda_rs")]
mod gpu {
    use cuda_rs::CublasHandle;
    use std::sync::OnceLock;

    static HANDLE: OnceLock<Option<CublasHandle>> = OnceLock::new();

    pub(super) fn handle() -> Option<&'static CublasHandle> {
        HANDLE.get_or_init(|| CublasHandle::new().ok()).as_ref()
    }
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct BitNetConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
}

impl Default for BitNetConfig {
    fn default() -> Self {
        Self {
            hidden_size: 2560,
            num_hidden_layers: 30,
            num_attention_heads: 20,
            num_key_value_heads: 5,
            intermediate_size: 6912,
            vocab_size: 128256,
            max_position_embeddings: 4096,
            rope_theta: 500_000.0,
            rms_norm_eps: 1e-5,
        }
    }
}

// ---------------------------------------------------------------------------
// RMSNorm
// ---------------------------------------------------------------------------

pub struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    pub fn new(weight: Tensor, eps: f64) -> Self {
        Self { weight, eps }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_f32 = x.to_dtype(DType::F32)?;
        let rms = (x_f32.sqr()?.mean_keepdim(D::Minus1)? + self.eps)?.sqrt()?;
        let normed = x_f32.broadcast_div(&rms)?;
        let w = self.weight.to_device(x.device())?.to_dtype(DType::F32)?;
        normed.broadcast_mul(&w).map_err(Into::into)
    }
}

// ---------------------------------------------------------------------------
// BitLinear — online W1.58A8 quantization matching the training regime
// ---------------------------------------------------------------------------

pub(crate) fn bitlinear_forward(x: &Tensor, w: &Tensor) -> Result<Tensor> {
    #[cfg(feature = "cuda_rs")]
    if let Some(h) = gpu::handle() {
        return bitlinear_gpu(x, w, h);
    }
    bitlinear_cpu(x, w)
}

fn bitlinear_cpu(x: &Tensor, w: &Tensor) -> Result<Tensor> {
    let w_f = w.to_dtype(DType::F32)?;
    let w_scale = w_f.abs()?.mean_all()?.clamp(1e-8_f64, f64::MAX)?;
    let w_q = w_f.broadcast_div(&w_scale)?.round()?.clamp(-1.0_f64, 1.0_f64)?;

    let x_f = x.to_dtype(DType::F32)?;
    let x_scale = x_f
        .abs()?
        .max(D::Minus1)?
        .unsqueeze(D::Minus1)?
        .clamp(1e-8_f64, f64::MAX)?
        .affine(1.0 / 127.0, 0.0)?;
    let x_q = x_f.broadcast_div(&x_scale)?.round()?.clamp(-128.0_f64, 127.0_f64)?;

    let out = x_q.matmul(&w_q.t()?)?.broadcast_mul(&x_scale)?.broadcast_mul(&w_scale)?;
    Ok(out.to_dtype(x.dtype())?)
}

/// GPU-accelerated BitLinear matmul via cuBLAS INT8.
///
/// Quantises weights (absmean → ternary i8) and activations (absmax per-token
/// → i8) on CPU, ships them to GPU, runs `cublasGemmEx` (INT8→I32), then
/// rescales the I32 result back to F32 on CPU.
#[cfg(feature = "cuda_rs")]
fn bitlinear_gpu(x: &Tensor, w: &Tensor, handle: &cuda_rs::CublasHandle) -> Result<Tensor> {
    use cuda_rs::DeviceBuffer;

    let orig_shape = x.dims().to_vec();
    let k = *orig_shape.last().unwrap();
    let m: usize = orig_shape[..orig_shape.len() - 1].iter().product();

    // ── Weight quantization (CPU) ────────────────────────────────────────────
    let w_f: Vec<f32> = w.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
    let (n, _k) = w.dims2()?; // w is [out_features, in_features]
    let w_abs_sum: f32 = w_f.iter().map(|v| v.abs()).sum::<f32>();
    let w_scale = (w_abs_sum / w_f.len() as f32).max(1e-8_f32);

    // Quantise and transpose w to [K, N] row-major (required by matmul_i8_i32)
    let mut w_qt: Vec<i8> = vec![0i8; k * n];
    for row in 0..n {
        for col in 0..k {
            let v = (w_f[row * k + col] / w_scale).round().clamp(-1.0, 1.0);
            w_qt[col * n + row] = v as i8;
        }
    }
    let w_gpu = DeviceBuffer::<i8>::from_slice(&w_qt)
        .map_err(|e| anyhow::anyhow!("w upload: {e}"))?;

    // ── Activation quantization (CPU) ────────────────────────────────────────
    let x_f: Vec<f32> = x.to_dtype(DType::F32)?.reshape((m, k))?.flatten_all()?.to_vec1()?;
    let x_scales: Vec<f32> = x_f.chunks(k)
        .map(|row| row.iter().cloned().fold(0.0_f32, f32::max).max(1e-8_f32) / 127.0)
        .collect();

    let x_qi8: Vec<i8> = x_f.chunks(k).zip(x_scales.iter())
        .flat_map(|(row, &s)| row.iter().map(move |&v| (v / s).round().clamp(-128.0, 127.0) as i8))
        .collect();
    let x_gpu = DeviceBuffer::<i8>::from_slice(&x_qi8)
        .map_err(|e| anyhow::anyhow!("x upload: {e}"))?;

    // ── cuBLAS INT8 matmul → I32 ─────────────────────────────────────────────
    let mut out_gpu = DeviceBuffer::<i32>::uninit(m * n)
        .map_err(|e| anyhow::anyhow!("out alloc: {e}"))?;
    handle.matmul_i8_i32(m, n, k, &x_gpu, &w_gpu, &mut out_gpu)
        .map_err(|e| anyhow::anyhow!("matmul: {e}"))?;
    let out_i32 = out_gpu.to_vec()
        .map_err(|e| anyhow::anyhow!("out download: {e}"))?;

    // ── Rescale I32 → F32, restore original shape ────────────────────────────
    let out_f32: Vec<f32> = out_i32.chunks(n).zip(x_scales.iter())
        .flat_map(|(row, &xs)| row.iter().map(move |&v| v as f32 * xs * w_scale))
        .collect();

    let flat = Tensor::from_vec(out_f32, (m, n), x.device())?;
    let out = flat.to_dtype(x.dtype())?;

    if orig_shape.len() > 2 {
        let mut out_shape = orig_shape;
        *out_shape.last_mut().unwrap() = n;
        return Ok(out.reshape(out_shape)?);
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Rotary embeddings (RoPE)
// ---------------------------------------------------------------------------

pub struct RotaryEmbedding {
    cos: Tensor,
    sin: Tensor,
}

impl RotaryEmbedding {
    pub fn new(head_dim: usize, max_len: usize, theta: f64, device: &Device) -> Result<Self> {
        let half = head_dim / 2;
        let inv_freq: Vec<f32> = (0..half)
            .map(|i| 1.0_f32 / (theta as f32).powf(i as f32 * 2.0 / head_dim as f32))
            .collect();
        let inv_freq = Tensor::from_vec(inv_freq, (1, half), device)?.to_dtype(DType::F32)?;
        let positions = Tensor::arange(0u32, max_len as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((max_len, 1))?;
        let freqs = positions.matmul(&inv_freq)?; // [max_len, half]
        let emb = Tensor::cat(&[&freqs, &freqs], D::Minus1)?; // [max_len, head_dim]
        Ok(Self {
            cos: emb.cos()?,
            sin: emb.sin()?,
        })
    }

    pub fn apply(&self, q: &Tensor, k: &Tensor, offset: usize) -> Result<(Tensor, Tensor)> {
        let seq = q.dim(2)?;
        let cos = self.cos.narrow(0, offset, seq)?.to_device(q.device())?.unsqueeze(0)?.unsqueeze(0)?;
        let sin = self.sin.narrow(0, offset, seq)?.to_device(q.device())?.unsqueeze(0)?.unsqueeze(0)?;
        Ok((rotate_half(q, &cos, &sin)?, rotate_half(k, &cos, &sin)?))
    }
}

fn rotate_half(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let half = x.dim(D::Minus1)? / 2;
    let x1 = x.narrow(D::Minus1, 0, half)?;
    let x2 = x.narrow(D::Minus1, half, half)?;
    let rotated = Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)?;
    Ok((x.broadcast_mul(cos)? + rotated.broadcast_mul(sin)?)?)
}

// ---------------------------------------------------------------------------
// Attention (with grouped-query attention and LoRA adapters)
// ---------------------------------------------------------------------------

pub struct BitNetAttention {
    q_proj: LoraLinear,
    k_proj: LoraLinear,
    v_proj: LoraLinear,
    o_proj: LoraLinear,
    attn_sub_norm: RmsNorm,
    head_dim: usize,
    num_heads: usize,
    num_kv_heads: usize,
    rope: RotaryEmbedding,
}

impl BitNetAttention {
    pub fn new(
        weights: &HashMap<String, Tensor>,
        prefix: &str,
        cfg: &BitNetConfig,
        lora_rank: usize,
        lora_alpha: f64,
        device: &Device,
    ) -> Result<Self> {
        let head_dim = cfg.hidden_size / cfg.num_attention_heads;

        let w = |name: &str| -> Result<Tensor> {
            weights
                .get(&format!("{prefix}.{name}.weight"))
                .with_context(|| format!("missing weight {prefix}.{name}.weight"))
                .cloned()
        };

        let q_proj = LoraLinear::new(w("self_attn.q_proj")?, lora_rank, lora_alpha, device)?;
        let k_proj = LoraLinear::new(w("self_attn.k_proj")?, lora_rank, lora_alpha, device)?;
        let v_proj = LoraLinear::new(w("self_attn.v_proj")?, lora_rank, lora_alpha, device)?;
        let o_proj = LoraLinear::new(w("self_attn.o_proj")?, lora_rank, lora_alpha, device)?;
        let attn_sub_norm = RmsNorm::new(w("self_attn.attn_sub_norm")?, cfg.rms_norm_eps);

        let rope = RotaryEmbedding::new(
            head_dim,
            cfg.max_position_embeddings,
            cfg.rope_theta,
            device,
        )?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            attn_sub_norm,
            head_dim,
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            rope,
        })
    }

    pub fn forward(&self, x: &Tensor, offset: usize, mask: Option<&Tensor>) -> Result<Tensor> {
        let (b, seq, _) = x.dims3()?;

        let q = self.q_proj.forward(x)?
            .reshape((b, seq, self.num_heads, self.head_dim))?
            .transpose(1, 2)?; // [b, heads, seq, head_dim]

        let k = self.k_proj.forward(x)?
            .reshape((b, seq, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let v = self.v_proj.forward(x)?
            .reshape((b, seq, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let (q, k) = self.rope.apply(&q, &k, offset)?;

        // Repeat k/v for GQA (20 heads / 5 kv-heads = 4 repeats)
        let repeat = self.num_heads / self.num_kv_heads;
        let k = k.repeat((1, repeat, 1, 1))?;
        let v = v.repeat((1, repeat, 1, 1))?;

        let scale = (self.head_dim as f64).sqrt().recip();
        let mut attn = q.matmul(&k.transpose(D::Minus2, D::Minus1)?)?.affine(scale, 0.0)?;

        if let Some(m) = mask {
            attn = attn.broadcast_add(m)?;
        }

        let attn = candle_nn::ops::softmax(&attn, D::Minus1)?;
        let out = attn
            .matmul(&v)?
            .transpose(1, 2)?
            .reshape((b, seq, self.num_heads * self.head_dim))?;
        let out = self.attn_sub_norm.forward(&out)?;
        self.o_proj.forward(&out).map_err(Into::into)
    }

    pub fn lora_vars(&self) -> Vec<&candle_core::Var> {
        let mut v = Vec::new();
        for arr in [
            self.q_proj.vars(),
            self.k_proj.vars(),
            self.v_proj.vars(),
            self.o_proj.vars(),
        ] {
            v.extend(arr);
        }
        v
    }
}

// ---------------------------------------------------------------------------
// MLP (gate-up-down with relu² activation)
// ---------------------------------------------------------------------------

pub struct BitNetMlp {
    gate_proj: Tensor,
    up_proj: Tensor,
    down_proj: Tensor,
    ffn_sub_norm: RmsNorm,
}

impl BitNetMlp {
    pub fn new(weights: &HashMap<String, Tensor>, prefix: &str, cfg: &BitNetConfig) -> Result<Self> {
        let w = |name: &str| -> Result<Tensor> {
            weights
                .get(&format!("{prefix}.{name}.weight"))
                .with_context(|| format!("missing {prefix}.{name}.weight"))
                .cloned()
        };
        let ffn_sub_norm = RmsNorm::new(w("mlp.ffn_sub_norm")?, cfg.rms_norm_eps);
        Ok(Self {
            gate_proj: w("mlp.gate_proj")?,
            up_proj: w("mlp.up_proj")?,
            down_proj: w("mlp.down_proj")?,
            ffn_sub_norm,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_shape = x.dims().to_vec();
        let in_dim = *x_shape.last().unwrap();
        let leading: usize = x_shape[..x_shape.len() - 1].iter().product();
        let flat = x.reshape((leading, in_dim))?;

        let gate = bitlinear_forward(&flat, &self.gate_proj)?.relu()?.sqr()?;
        let up   = bitlinear_forward(&flat, &self.up_proj)?;
        let hidden = (gate * up)?;
        let hidden = self.ffn_sub_norm.forward(&hidden)?;
        let out = bitlinear_forward(&hidden, &self.down_proj)?;

        let out_features = out.dim(1)?;
        let mut out_shape = x_shape;
        *out_shape.last_mut().unwrap() = out_features;
        out.reshape(out_shape).map_err(Into::into)
    }
}

// ---------------------------------------------------------------------------
// Decoder layer
// ---------------------------------------------------------------------------

pub struct BitNetDecoderLayer {
    attn: BitNetAttention,
    mlp: BitNetMlp,
    input_norm: RmsNorm,
    post_attn_norm: RmsNorm,
}

impl BitNetDecoderLayer {
    pub fn new(
        weights: &HashMap<String, Tensor>,
        layer_idx: usize,
        cfg: &BitNetConfig,
        lora_rank: usize,
        lora_alpha: f64,
        device: &Device,
    ) -> Result<Self> {
        let prefix = format!("model.layers.{layer_idx}");
        let w = |name: &str| -> Result<Tensor> {
            weights
                .get(&format!("{prefix}.{name}"))
                .with_context(|| format!("missing {prefix}.{name}"))
                .cloned()
        };

        Ok(Self {
            attn: BitNetAttention::new(weights, &prefix, cfg, lora_rank, lora_alpha, device)?,
            mlp: BitNetMlp::new(weights, &prefix, cfg)?,
            input_norm: RmsNorm::new(w("input_layernorm.weight")?, cfg.rms_norm_eps),
            post_attn_norm: RmsNorm::new(w("post_attention_layernorm.weight")?, cfg.rms_norm_eps),
        })
    }

    pub fn forward(&self, x: &Tensor, offset: usize, mask: Option<&Tensor>) -> Result<Tensor> {
        let residual = x;
        let x = self.input_norm.forward(x)?;
        let x = self.attn.forward(&x, offset, mask)?;
        let x = (residual + x)?;
        let residual = &x;
        let x = self.post_attn_norm.forward(&x)?;
        let x = self.mlp.forward(&x)?;
        Ok((residual + x)?)
    }

    pub fn lora_vars(&self) -> Vec<&candle_core::Var> {
        self.attn.lora_vars()
    }
}

// ---------------------------------------------------------------------------
// Full model
// ---------------------------------------------------------------------------

pub struct BitNetModel {
    embed: Tensor,
    layers: Vec<BitNetDecoderLayer>,
    norm: RmsNorm,
    lm_head: Tensor,
}

impl BitNetModel {
    pub fn new(
        weights: &HashMap<String, Tensor>,
        cfg: &BitNetConfig,
        lora_rank: usize,
        lora_alpha: f64,
        device: &Device,
    ) -> Result<Self> {
        let w = |name: &str| -> Result<Tensor> {
            weights
                .get(name)
                .with_context(|| format!("missing weight {name}"))
                .cloned()
        };

        let embed = w("model.embed_tokens.weight")?;
        let norm = RmsNorm::new(w("model.norm.weight")?, cfg.rms_norm_eps);
        let lm_head = embed.clone(); // BitNet ties word embeddings

        let layers = (0..cfg.num_hidden_layers)
            .map(|i| BitNetDecoderLayer::new(weights, i, cfg, lora_rank, lora_alpha, device))
            .collect::<Result<Vec<_>>>()?;

        Ok(Self { embed, layers, norm, lm_head })
    }

    /// Forward pass. Returns logits `[batch, seq, vocab]`.
    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let (b, seq) = input_ids.dims2()?;
        let flat_ids = input_ids.reshape((b * seq,))?;
        let mut x = self.embed
            .index_select(&flat_ids, 0)?
            .reshape((b, seq, self.embed.dim(1)?))?
            .to_dtype(DType::F32)?;

        let mask = causal_mask(seq, x.dtype(), x.device())?;
        for layer in &self.layers {
            x = layer.forward(&x, 0, Some(&mask))?;
        }

        x = self.norm.forward(&x)?;
        let (b, seq, hidden) = x.dims3()?;
        let flat = x.reshape((b * seq, hidden))?;
        // lm_head stored in BF16 (655 MB vs 1.31 GB in F32) to keep within 4 GB VRAM budget
        let flat_bf16 = flat.to_dtype(DType::BF16)?;
        let logits_flat = flat_bf16.matmul(&self.lm_head.t()?)?.to_dtype(DType::F32)?;
        let vocab = logits_flat.dim(1)?;
        logits_flat.reshape((b, seq, vocab)).map_err(Into::into)
    }

    /// Collect all LoRA Vars from every layer (used by the optimizer).
    pub fn lora_vars(&self) -> Vec<&candle_core::Var> {
        self.layers.iter().flat_map(|l| l.lora_vars()).collect()
    }
}

fn causal_mask(seq: usize, dtype: DType, device: &Device) -> Result<Tensor> {
    let mask: Vec<f32> = (0..seq)
        .flat_map(|row| {
            (0..seq).map(move |col| if col > row { f32::NEG_INFINITY } else { 0.0 })
        })
        .collect();
    Tensor::from_vec(mask, (seq, seq), device)?
        .to_dtype(dtype)
        .map_err(Into::into)
}
