use std::{io::Write as _};
use std::sync::Arc;

use tokio::sync::watch;

use bitnet_rs::{run_inference, ChatMessage, InferenceConfig};

// Data  structure size in bytes
/* 
fn ds_size() {
    use std::mem::size_of;
    eprintln!("Sizes:");
    eprintln!("  String: {} bytes", size_of::<Arc<NulError>>());
    eprintln!("  InferenceConfig: {} bytes", size_of::<InferenceConfig>());
}
*/
fn main() {

    // Derive workspace root from executable location:
    // target/{profile}/bitnet-cli.exe → parent → parent → parent → workspace root
    let mut model = std::env::current_exe()
        .ok()
        .and_then(|p| {
            p.parent()?.parent()?.parent()
                .map(|root| root.join("models").join("bitnet-b1.58-2b-4t-bf16").to_string_lossy().into_owned())
        })
        .unwrap_or_else(|| "./models/bitnet-b1.58-2b-4t-bf16".to_string());
    let mut prompt      = String::new();
    let mut system      = String::new();
    let mut temperature    = 0.6_f64;
    let mut top_p          = 0.9_f64;
    let mut max_new_tokens = 256_usize;

    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--model"          => { model          = args[i + 1].clone(); i += 2; }
            "--prompt"         => { prompt          = args[i + 1].clone(); i += 2; }
            "--system"         => { system          = args[i + 1].clone(); i += 2; }
            "--temperature"    => { temperature    = args[i + 1].parse().unwrap_or(0.6); i += 2; }
            "--top-p"          => { top_p          = args[i + 1].parse().unwrap_or(0.9); i += 2; }
            "--max-new-tokens" => { max_new_tokens  = args[i + 1].parse().unwrap_or(512);  i += 2; }
            other => { eprintln!("Unknown arg: {other}"); i += 1; }
        }
    }

    let on_token: Arc<dyn Fn(String) + Send + Sync> =
        Arc::new(|t| { print!("{t}"); let _ = std::io::stdout().flush(); });

    let on_log: Arc<dyn Fn(String) + Send + Sync> =
        Arc::new(|s| eprintln!("{s}"));

    let (_, cancel_rx) = watch::channel(false);

    let mut base_messages: Vec<ChatMessage> = Vec::new();
    if !system.is_empty() {
        base_messages.push(ChatMessage { role: "system".into(), content: system });
    }

    if !prompt.is_empty() {
        let mut messages = base_messages;
        messages.push(ChatMessage { role: "user".into(), content: prompt });
        eprint!("\nAssistant: ");
        let cfg = InferenceConfig {
            model_path: model,
            messages,
            max_new_tokens,
            temperature,
            top_p,
        };
        match run_inference(cfg, cancel_rx, on_token, on_log) {
            Ok(_) => println!(),
            Err(e) => eprintln!("Error: {e}"),
        }
        return;
    }

    // Interactive chat loop
    eprintln!("BitNet b1.58 2B — type 'exit' or Ctrl-C to quit.");
    let mut history = base_messages;
    loop {
        eprint!("\nYou: ");
        let _ = std::io::stderr().flush();
        let mut line = String::new();
        if std::io::stdin().read_line(&mut line).is_err() {
            break;
        }
        let line = line.trim().to_string();
        if line.is_empty() || line.eq_ignore_ascii_case("exit") {
            break;
        }
        history.push(ChatMessage { role: "user".into(), content: line });

        eprint!("\nAssistant: ");
        let _ = std::io::stderr().flush();

        let cfg = InferenceConfig {
            model_path: model.clone(),
            messages: history.clone(),
            max_new_tokens,
            temperature,
            top_p,
        };
        match run_inference(cfg, cancel_rx.clone(), on_token.clone(), on_log.clone()) {
            Ok(reply) => {
                println!();
                history.push(ChatMessage { role: "assistant".into(), content: reply });
            }
            Err(e) => {
                eprintln!("Error: {e}");
                break;
            }
        }
    }
}
