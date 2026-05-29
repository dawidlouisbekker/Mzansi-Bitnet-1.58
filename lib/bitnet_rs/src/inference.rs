use std::path::Path;
use std::sync::Arc;

use anyhow::Result;
use candle_core::{Tensor, D};
use rand::distributions::{Distribution, WeightedIndex};
use rand::{SeedableRng, rngs::StdRng};
use tokio::sync::watch;

use crate::bitnet::{BitNetConfig, BitNetModel};
use crate::mmap::load_mmap;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct InferenceConfig {
    pub model_path: String,
    pub messages: Vec<ChatMessage>,
    pub max_new_tokens: usize,
    pub temperature: f64,
    pub top_p: f64,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            model_path: "./models/bitnet-b1.58-2b-4t-bf16".into(),
            messages: Vec::new(),
            max_new_tokens: 256,
            temperature: 0.6,
            top_p: 0.9,
        }
    }
}

pub type TokenCallback = Arc<dyn Fn(String) + Send + Sync>;
pub type InferenceLogCallback = Arc<dyn Fn(String) + Send + Sync>;

// ---------------------------------------------------------------------------
// Chat template — matches tokenizer_config.json:
//   "{Role}: {content}<|eot_id|>" for each turn, then "Assistant: "
// ---------------------------------------------------------------------------

fn capitalize_first(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        None => String::new(),
        Some(f) => f.to_uppercase().to_string() + c.as_str(),
    }
}

fn format_prompt(messages: &[ChatMessage]) -> String {
    let mut prompt = String::new();
    for msg in messages {
        let role = capitalize_first(&msg.role);
        prompt.push_str(&format!("{}: {}<|eot_id|>", role, msg.content.trim()));
    }
    prompt.push_str("Assistant: ");
    prompt
}

// ---------------------------------------------------------------------------
// Token sampling
// ---------------------------------------------------------------------------

fn sample_next(logits: &Tensor, temperature: f64, top_p: f64, rng: &mut StdRng) -> Result<u32> {
    if temperature <= 0.0 {
        return Ok(logits.argmax(D::Minus1)?.to_scalar::<u32>()?);
    }
    let scaled = logits.affine(1.0 / temperature, 0.0)?;
    let probs = candle_nn::ops::softmax(&scaled.unsqueeze(0)?, D::Minus1)?;
    let mut probs_vec: Vec<f32> = probs.squeeze(0)?.to_vec1()?;

    // Top-p (nucleus) filtering
    if top_p < 1.0 {
        let mut indexed: Vec<(usize, f32)> = probs_vec.iter().copied().enumerate().collect();
        indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let mut cumsum = 0.0_f32;
        for (idx, p) in &indexed {
            cumsum += p;
            if cumsum - p >= top_p as f32 {
                probs_vec[*idx] = 0.0;
            }
        }
    }

    let probs_clamped: Vec<f32> = probs_vec.iter().map(|&p| p.max(0.0)).collect();
    let dist = WeightedIndex::new(&probs_clamped)
        .map_err(|e| anyhow::anyhow!("sampling distribution error: {e}"))?;
    Ok(dist.sample(rng) as u32)
}

// ---------------------------------------------------------------------------
// Main inference entry point
// ---------------------------------------------------------------------------

/// Run autoregressive generation. Streams decoded text fragments via `on_token`.
/// Blocks until `max_new_tokens` is reached, the stop marker is generated, or
/// `cancel_rx` fires.
pub fn run_inference(
    cfg: InferenceConfig,
    cancel_rx: watch::Receiver<bool>,
    on_token: TokenCallback,
    on_log: InferenceLogCallback,
) -> Result<String> {
    let device = crate::utils::pick_device(&|s| on_log(s));
    let model_dir = Path::new(&cfg.model_path);

    // ── 1. Load weights ───────────────────────────────────────────────────
    let weights_path = model_dir.join("model.safetensors");
    on_log(format!("Loading weights from {}…", weights_path.display()));
    let mmap = load_mmap(&weights_path, &device)?;

    let bitnet_cfg = BitNetConfig::default();
    // rank=1 / alpha=1 → LoRA B is zero-initialised so contribution is zero;
    // effectively runs the base weights only.
    let model = BitNetModel::new(&mmap.tensors, &bitnet_cfg, 1, 1.0, &candle_core::Device::Cpu)?;
    on_log("Model ready".into());

    // ── 2. Load tokenizer ─────────────────────────────────────────────────
    let tokenizer_path = model_dir.join("tokenizer.json");
    let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("tokenizer load error: {e}"))?;
    on_log("Tokenizer loaded".into());

    // ── 3. Tokenize prompt ────────────────────────────────────────────────
    const BOS_ID: u32 = 128_000;
    const EOS_IDS: &[u32] = &[128_001, 128_009]; // <|end_of_text|> and <|eot_id|>

    let prompt = format_prompt(&cfg.messages);
    let encoding = tokenizer
        .encode(prompt.as_str(), false)
        .map_err(|e| anyhow::anyhow!("tokenizer encode error: {e}"))?;
    let mut token_ids: Vec<u32> = std::iter::once(BOS_ID)
        .chain(encoding.get_ids().iter().copied())
        .collect();
    on_log(format!("Prompt: {} tokens. Generating…", token_ids.len()));

    // ── 4. Generation loop ────────────────────────────────────────────────
    let mut rng = StdRng::from_entropy();
    let mut generated_ids: Vec<u32> = Vec::new();
    let mut prev_streamed_len: usize = 0;
    let mut full_decoded = String::new();

    for _ in 0..cfg.max_new_tokens {
        if *cancel_rx.borrow() {
            on_log("Generation cancelled.".into());
            break;
        }

        let seq_len = token_ids.len();
        let input = Tensor::from_vec(token_ids.clone(), (1, seq_len), &device)?;
        let logits = model.forward(&input)?; // [1, seq, vocab]
        let vocab = logits.dim(2)?;

        // Slice last-position logits → [vocab]
        let last_logits = logits.narrow(1, seq_len - 1, 1)?.reshape((vocab,))?;

        let next_id = sample_next(&last_logits, cfg.temperature, cfg.top_p, &mut rng)?;

        if EOS_IDS.contains(&next_id) {
            break;
        }

        generated_ids.push(next_id);
        token_ids.push(next_id);

        // Decode and stream incrementally
        full_decoded = tokenizer
            .decode(&generated_ids, true)
            .map_err(|e| anyhow::anyhow!("decode error: {e}"))?;

        let safe_end = char_boundary_floor(&full_decoded, full_decoded.len());
        if safe_end > prev_streamed_len {
            on_token(full_decoded[prev_streamed_len..safe_end].to_string());
            prev_streamed_len = safe_end;
        }
    }

    Ok(full_decoded)
}

// Walk backwards from `pos` to find the nearest valid UTF-8 char boundary.
fn char_boundary_floor(s: &str, pos: usize) -> usize {
    let mut p = pos.min(s.len());
    while p > 0 && !s.is_char_boundary(p) {
        p -= 1;
    }
    p
}
