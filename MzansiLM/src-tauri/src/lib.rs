use std::sync::{Arc, Mutex};

use bitnet_rs::{
    run_inference, run_training, InferenceConfig, InferenceLogCallback,
    LogCallback, ProgressEvent, SampleCallback, TokenCallback, TrainingConfig,
};
use tauri::{AppHandle, Emitter, Manager, State};
use tokio::sync::watch;

// ---------------------------------------------------------------------------
// Shared state
// ---------------------------------------------------------------------------

struct TrainingState {
    cancel_tx: Mutex<Option<watch::Sender<bool>>>,
    status: Mutex<Option<ProgressEvent>>,
}

impl Default for TrainingState {
    fn default() -> Self {
        Self {
            cancel_tx: Mutex::new(None),
            status: Mutex::new(None),
        }
    }
}

// ---------------------------------------------------------------------------
// Events
// ---------------------------------------------------------------------------

#[derive(Clone, serde::Serialize)]
struct SampleEvent {
    epoch: usize,
    sample_idx: usize,
    text: String,
}

// ---------------------------------------------------------------------------
// Inference state
// ---------------------------------------------------------------------------

struct InferenceState {
    cancel_tx: Mutex<Option<watch::Sender<bool>>>,
}

impl Default for InferenceState {
    fn default() -> Self {
        Self {
            cancel_tx: Mutex::new(None),
        }
    }
}

// ---------------------------------------------------------------------------
// Inference events
// ---------------------------------------------------------------------------

#[derive(Clone, serde::Serialize)]
struct TokenEvent {
    text: String,
}

// ---------------------------------------------------------------------------
// Training commands
// ---------------------------------------------------------------------------

#[tauri::command]
async fn start_training(
    config: TrainingConfig,
    state: State<'_, TrainingState>,
    app: AppHandle,
) -> Result<(), String> {
    if state.cancel_tx.lock().unwrap().is_some() {
        return Err("Training is already running".into());
    }

    let (cancel_tx, cancel_rx) = watch::channel(false);
    *state.cancel_tx.lock().unwrap() = Some(cancel_tx);

    let app_progress = app.clone();
    let app_sample = app.clone();
    let app_log = app.clone();

    let on_progress: bitnet_rs::ProgressCallback = Arc::new(move |event: ProgressEvent| {
        let _ = app_progress.emit("training://progress", &event);
    });

    let on_sample: SampleCallback = Arc::new(move |epoch, sample_idx, text| {
        let _ = app_sample.emit("training://sample", &SampleEvent { epoch, sample_idx, text });
    });

    let on_log: LogCallback = Arc::new(move |msg: String| {
        let _ = app_log.emit("training://log", &msg);
    });

    tauri::async_runtime::spawn(async move {
        let result = tokio::task::spawn_blocking(move || {
            run_training(config, cancel_rx, on_progress, on_sample, on_log)
        })
        .await;

        let _ = app.emit(
            "training://done",
            match result {
                Ok(Ok(())) => serde_json::json!({"success": true}),
                Ok(Err(e)) => serde_json::json!({"success": false, "error": e.to_string()}),
                Err(e) => serde_json::json!({"success": false, "error": e.to_string()}),
            },
        );

        if let Some(window) = app.get_webview_window("main") {
            if let Some(s) = window.app_handle().try_state::<TrainingState>() {
                *s.cancel_tx.lock().unwrap() = None;
            }
        }
    });

    Ok(())
}

#[tauri::command]
fn stop_training(state: State<'_, TrainingState>) -> Result<(), String> {
    if let Some(tx) = state.cancel_tx.lock().unwrap().take() {
        let _ = tx.send(true);
        Ok(())
    } else {
        Err("No training is running".into())
    }
}

#[tauri::command]
fn get_training_status(state: State<'_, TrainingState>) -> Option<ProgressEvent> {
    state.status.lock().unwrap().clone()
}

// ---------------------------------------------------------------------------
// Inference commands
// ---------------------------------------------------------------------------

#[tauri::command]
async fn start_inference(
    config: InferenceConfig,
    state: State<'_, InferenceState>,
    app: AppHandle,
) -> Result<(), String> {
    if state.cancel_tx.lock().unwrap().is_some() {
        return Err("Inference is already running".into());
    }

    let (cancel_tx, cancel_rx) = watch::channel(false);
    *state.cancel_tx.lock().unwrap() = Some(cancel_tx);

    let app_token = app.clone();
    let app_log = app.clone();

    let on_token: TokenCallback = Arc::new(move |text: String| {
        let _ = app_token.emit("inference://token", &TokenEvent { text });
    });

    let on_log: InferenceLogCallback = Arc::new(move |msg: String| {
        let _ = app_log.emit("inference://log", &msg);
    });

    tauri::async_runtime::spawn(async move {
        let result = tokio::task::spawn_blocking(move || {
            run_inference(config, cancel_rx, on_token, on_log)
        })
        .await;

        let _ = app.emit(
            "inference://done",
            match result {
                Ok(Ok(_)) => serde_json::json!({ "success": true }),
                Ok(Err(e)) => serde_json::json!({ "success": false, "error": e.to_string() }),
                Err(e) => serde_json::json!({ "success": false, "error": e.to_string() }),
            },
        );

        if let Some(window) = app.get_webview_window("main") {
            if let Some(s) = window.app_handle().try_state::<InferenceState>() {
                *s.cancel_tx.lock().unwrap() = None;
            }
        }
    });

    Ok(())
}

#[tauri::command]
fn stop_inference(state: State<'_, InferenceState>) -> Result<(), String> {
    if let Some(tx) = state.cancel_tx.lock().unwrap().take() {
        let _ = tx.send(true);
        Ok(())
    } else {
        Err("No inference is running".into())
    }
}

// ---------------------------------------------------------------------------
// File commands
// ---------------------------------------------------------------------------

#[tauri::command]
async fn read_file(path: String) -> Result<String, String> {
    tokio::fs::read_to_string(&path).await.map_err(|e| e.to_string())
}

#[tauri::command]
async fn write_file(path: String, content: String) -> Result<(), String> {
    if let Some(parent) = std::path::Path::new(&path).parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .map_err(|e| e.to_string())?;
    }
    tokio::fs::write(&path, content).await.map_err(|e| e.to_string())
}

// ---------------------------------------------------------------------------
// App entry point
// ---------------------------------------------------------------------------

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .manage(TrainingState::default())
        .manage(InferenceState::default())
        .invoke_handler(tauri::generate_handler![
            start_training,
            stop_training,
            get_training_status,
            start_inference,
            stop_inference,
            read_file,
            write_file,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
