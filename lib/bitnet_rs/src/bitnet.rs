/// BitNet b1.58 2B architecture implemented with candle.
///
/// Config matches ./models/bitnet-b1.58-2b-4t-bf16/config.json:
///   hidden_size=2560, num_hidden_layers=30, num_attention_heads=20,
///   num_key_value_heads=5, intermediate_size=6912, rope_theta=500000.
use std::collections::HashMap;

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor, D};

use crate::lora::LoraLinear;

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
        normed
            .broadcast_mul(&self.weight.to_dtype(DType::F32)?)
            .map_err(Into::into)
    }
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
        let cos = self.cos.narrow(0, offset, seq)?.unsqueeze(0)?.unsqueeze(0)?;
        let sin = self.sin.narrow(0, offset, seq)?.unsqueeze(0)?.unsqueeze(0)?;
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
}

impl BitNetMlp {
    pub fn new(weights: &HashMap<String, Tensor>, prefix: &str) -> Result<Self> {
        let w = |name: &str| -> Result<Tensor> {
            weights
                .get(&format!("{prefix}.{name}.weight"))
                .with_context(|| format!("missing {prefix}.{name}.weight"))
                .cloned()
        };
        Ok(Self {
            gate_proj: w("mlp.gate_proj")?,
            up_proj: w("mlp.up_proj")?,
            down_proj: w("mlp.down_proj")?,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_shape = x.dims().to_vec();
        let in_dim = *x_shape.last().unwrap();
        let leading: usize = x_shape[..x_shape.len() - 1].iter().product();
        let flat = x.reshape((leading, in_dim))?.to_dtype(DType::F32)?;

        let gate = flat.matmul(&self.gate_proj.to_dtype(DType::F32)?.t()?)?.relu()?.sqr()?;
        let up = flat.matmul(&self.up_proj.to_dtype(DType::F32)?.t()?)?;
        let hidden = (gate * up)?;
        let out = hidden.matmul(&self.down_proj.to_dtype(DType::F32)?.t()?)?;

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
            mlp: BitNetMlp::new(weights, &prefix)?,
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
        // BitNet ties word embeddings — lm_head == embed_tokens
        let lm_head = embed.clone();

        let layers = (0..cfg.num_hidden_layers)
            .map(|i| {
                BitNetDecoderLayer::new(weights, i, cfg, lora_rank, lora_alpha, device)
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(Self { embed, layers, norm, lm_head })
    }

    /// Forward pass. Returns logits `[batch, seq, vocab]`.
    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let (b, seq) = input_ids.dims2()?;
        let flat_ids = input_ids.reshape((b * seq,))?;
        let mut x = self.embed.index_select(&flat_ids, 0)?
            .reshape((b, seq, self.embed.dim(1)?))?
            .to_dtype(DType::F32)?;

        // Causal mask
        let mask = causal_mask(seq, x.dtype(), x.device())?;

        for layer in &self.layers {
            x = layer.forward(&x, 0, Some(&mask))?;
        }

        x = self.norm.forward(&x)?;
        let (b, seq, hidden) = x.dims3()?;
        let flat = x.reshape((b * seq, hidden))?;
        let logits_flat = flat.matmul(&self.lm_head.to_dtype(DType::F32)?.t()?)?;
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
