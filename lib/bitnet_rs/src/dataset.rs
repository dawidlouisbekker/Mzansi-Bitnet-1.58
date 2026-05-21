use std::path::Path;

use anyhow::{Context, Result};
use serde::Deserialize;
use tokenizers::Tokenizer;

#[derive(Deserialize)]
struct Message {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct Example {
    messages: Vec<Message>,
}

pub struct Batch {
    /// Token IDs, shape `[seq_len]`
    pub input_ids: Vec<u32>,
    /// Same length as `input_ids`; positions to predict are token ids, masked positions are u32::MAX
    pub labels: Vec<u32>,
    /// Human-readable rendering of the messages, shown word-by-word in the UI
    pub text: String,
}

/// Load a JSONL dataset and tokenize every example.
///
/// Each line must be `{"messages": [{"role": "...", "content": "..."}]}`.
/// The assistant turn is used as the prediction target; other turns are masked.
pub fn load_dataset(path: &Path, tokenizer: &Tokenizer, max_len: usize) -> Result<Vec<Batch>> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("cannot read {}", path.display()))?;

    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|line| {
            let ex: Example =
                serde_json::from_str(line).context("JSONL parse error — expected {messages:[...]}")?;
            build_batch(ex, tokenizer, max_len)
        })
        .collect()
}

fn build_batch(ex: Example, tokenizer: &Tokenizer, max_len: usize) -> Result<Batch> {
    // Build a prompt string with a simple chat template
    let mut full_text = String::new();
    let mut assistant_spans: Vec<(usize, usize)> = Vec::new();

    for msg in &ex.messages {
        match msg.role.as_str() {
            "system" => {
                full_text.push_str("<|system|>\n");
                full_text.push_str(&msg.content);
                full_text.push_str("\n<|end|>\n");
            }
            "user" => {
                full_text.push_str("<|user|>\n");
                full_text.push_str(&msg.content);
                full_text.push_str("\n<|end|>\n<|assistant|>\n");
            }
            "assistant" => {
                let start = full_text.len();
                full_text.push_str(&msg.content);
                full_text.push_str("\n<|end|>\n");
                assistant_spans.push((start, full_text.len()));
            }
            _ => {}
        }
    }

    let text = ex.messages.iter()
        .map(|m| format!("[{}] {}", m.role, m.content))
        .collect::<Vec<_>>()
        .join("\n");

    let encoding = tokenizer
        .encode(full_text.as_str(), false)
        .map_err(|e| anyhow::anyhow!("tokenizer error: {e}"))?;

    let tokens: Vec<u32> = encoding.get_ids().to_vec();
    let offsets: Vec<(usize, usize)> = encoding.get_offsets().to_vec();

    // Mark which token positions correspond to assistant content
    let mut labels = vec![u32::MAX; tokens.len()]; // u32::MAX = masked
    for (char_start, char_end) in &assistant_spans {
        for (tok_idx, &(byte_start, byte_end)) in offsets.iter().enumerate() {
            if byte_start >= *char_start && byte_end <= *char_end {
                labels[tok_idx] = tokens[tok_idx];
            }
        }
    }

    // Truncate
    let len = tokens.len().min(max_len);
    Ok(Batch {
        input_ids: tokens[..len].to_vec(),
        labels: labels[..len].to_vec(),
        text,
    })
}
