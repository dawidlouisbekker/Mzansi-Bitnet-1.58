// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    // Set CWD to workspace root so relative paths like ./models/... resolve correctly.
    // CARGO_MANIFEST_DIR points to src-tauri/; workspace root is two levels up.
    let manifest_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest_dir.parent().unwrap().parent().unwrap();
    let _ = std::env::set_current_dir(workspace_root);

    mzansilm_lib::run()
}
