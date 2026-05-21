use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::{AdamW, Optimizer, ParamsAdamW};
use tokio::sync::watch;

use crate::bitnet::{BitNetConfig, BitNetModel};
use crate::dataset::{load_dataset, Batch};
use crate::mmap::load_mmap;

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct TrainingConfig {
    pub model_path: String,
    pub dataset_path: String,
    pub output_path: String,
    pub learning_rate: f64,
    pub epochs: usize,
    pub batch_size: usize,
    pub lora_rank: usize,
    pub lora_alpha: f64,
    pub grad_accum_steps: usize,
    pub max_seq_len: usize,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            model_path: "./models/bitnet-b1.58-2b-4t-bf16".into(),
            dataset_path: "./data/train.jsonl".into(),
            output_path: "./output".into(),
            learning_rate: 2e-4,
            epochs: 3,
            batch_size: 1,
            lora_rank: 16,
            lora_alpha: 32.0,
            grad_accum_steps: 4,
            max_seq_len: 2048,
        }
    }
}

#[derive(Clone, serde::Serialize)]
pub struct ProgressEvent {
    pub epoch: usize,
    pub step: usize,
    pub total_steps: usize,
    pub loss: f32,
}

pub type ProgressCallback = Arc<dyn Fn(ProgressEvent) + Send + Sync>;
pub type SampleCallback   = Arc<dyn Fn(usize, usize, String) + Send + Sync>;
pub type LogCallback      = Arc<dyn Fn(String) + Send + Sync>;

/// Run the training loop. Blocks until done or until `cancel_rx` fires.
pub fn run_training(
    cfg: TrainingConfig,
    cancel_rx: watch::Receiver<bool>,
    on_progress: ProgressCallback,
    on_sample: SampleCallback,
    on_log: LogCallback,
) -> Result<()> {
    let device = crate::utils::pick_device(&|s| on_log(s));
    let model_dir = Path::new(&cfg.model_path);

    // ── 1. Load weights via mmap ──────────────────────────────────────────
    let weights_path = model_dir.join("model.safetensors");
    on_log(format!("Loading weights from {}", weights_path.display()));
    let mmap = load_mmap(&weights_path, &device)?;

    let bitnet_cfg = BitNetConfig::default();
    let model = BitNetModel::new(
        &mmap.tensors,
        &bitnet_cfg,
        cfg.lora_rank,
        cfg.lora_alpha,
        &device,
    )?;
    on_log("Model ready".into());

    // ── 2. Load tokenizer ─────────────────────────────────────────────────
    let tokenizer_path = model_dir.join("tokenizer.json");
    let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("tokenizer load error: {e}"))?;
    on_log("Tokenizer loaded".into());

    // ── 3. Load dataset ───────────────────────────────────────────────────
    let dataset = load_dataset(
        Path::new(&cfg.dataset_path),
        &tokenizer,
        cfg.max_seq_len,
    )?;
    on_log(format!("Dataset: {} examples", dataset.len()));

    // ── 4. Optimiser over LoRA variables only ─────────────────────────────
    let lora_vars: Vec<candle_core::Var> = model.lora_vars().into_iter().cloned().collect();
    let mut optimizer = AdamW::new(
        lora_vars.clone(),
        ParamsAdamW {
            lr: cfg.learning_rate,
            weight_decay: 0.01,
            ..Default::default()
        },
    )?;

    let total_steps = cfg.epochs * dataset.len();
    let mut global_step = 0usize;

    // ── 5. Training loop ─────────────────────────────────────────────────
    for epoch in 0..cfg.epochs {
        on_log(format!("── Epoch {}/{} ──", epoch + 1, cfg.epochs));
        let mut accum_loss = Tensor::zeros((), DType::F32, &device)?;
        let mut accum_count = 0usize;

        for (batch_idx, batch) in dataset.iter().enumerate() {
            if *cancel_rx.borrow() {
                on_log(format!("Training cancelled at step {global_step}"));
                return Ok(());
            }

            on_sample(epoch + 1, batch_idx, batch.text.clone());

            let (input_ids, labels, valid_count) = prepare_batch(batch, &device)?;
            if valid_count == 0 {
                continue;
            }

            let logits = model.forward(&input_ids)?; // [1, seq, vocab]
            let loss = cross_entropy_loss(&logits, &labels, valid_count)?;

            // Gradient accumulation
            accum_loss = (accum_loss + &loss)?;
            accum_count += 1;

            if accum_count >= cfg.grad_accum_steps || batch_idx + 1 == dataset.len() {
                let avg_loss = accum_loss.affine(1.0 / accum_count as f64, 0.0)?;
                let grads = avg_loss.backward()?;
                optimizer.step(&grads)?;

                let loss_val: f32 = avg_loss.to_scalar()?;
                on_progress(ProgressEvent {
                    epoch: epoch + 1,
                    step: global_step + 1,
                    total_steps,
                    loss: loss_val,
                });
                on_log(format!(
                    "step {}/{} · loss {:.4}",
                    global_step + 1,
                    total_steps,
                    loss_val
                ));

                accum_loss = Tensor::zeros((), DType::F32, &device)?;
                accum_count = 0;
            }

            global_step += 1;
        }
    }

    // ── 6. Save LoRA adapter ─────────────────────────────────────────────
    let out_dir = PathBuf::from(&cfg.output_path);
    std::fs::create_dir_all(&out_dir)?;
    let out_path = out_dir.join("lora_adapter.safetensors");

    let named: std::collections::HashMap<String, Tensor> = lora_vars
        .iter()
        .enumerate()
        .flat_map(|(layer, var)| {
            [(format!("layer_{layer}.lora_a"), var.as_tensor().clone())]
        })
        .collect();

    candle_core::safetensors::save(&named, &out_path)?;
    on_log(format!("Adapter saved → {}", out_path.display()));
    Ok(())
}

fn prepare_batch(
    batch: &Batch,
    device: &Device,
) -> Result<(Tensor, Tensor, usize)> {
    let ids: Vec<u32> = batch.input_ids.clone();
    let labels_raw: Vec<u32> = batch.labels.clone();

    let valid_count = labels_raw.iter().filter(|&&l| l != u32::MAX).count();

    let input_ids = Tensor::from_vec(ids, (1, batch.input_ids.len()), device)?;
    let labels_clean: Vec<u32> = labels_raw
        .iter()
        .map(|&l| if l == u32::MAX { 0 } else { l })
        .collect();
    let labels = Tensor::from_vec(labels_clean, (1, batch.labels.len()), device)?;

    Ok((input_ids, labels, valid_count))
}

fn cross_entropy_loss(logits: &Tensor, labels: &Tensor, valid_count: usize) -> Result<Tensor> {
    // logits: [batch, seq, vocab]  labels: [batch, seq]
    let (b, seq, vocab) = logits.dims3()?;
    let logits_f = logits.to_dtype(DType::F32)?.reshape((b * seq, vocab))?;
    let labels_flat = labels.reshape((b * seq,))?;

    // Shift: predict next token (standard LM loss)
    let shift_logits = logits_f.narrow(0, 0, b * seq - 1)?;
    let shift_labels = labels_flat.narrow(0, 1, b * seq - 1)?;

    let log_softmax = candle_nn::ops::log_softmax(&shift_logits, 1)?;
    let gathered = log_softmax.gather(&shift_labels.unsqueeze(1)?, 1)?.squeeze(1)?;
    let neg_log = gathered.neg()?;
    Ok(neg_log.sum_all()?.affine(1.0 / valid_count.max(1) as f64, 0.0)?)
}
