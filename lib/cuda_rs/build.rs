use std::path::PathBuf;

const CUDA_VARS: &[&str] = &["CUDA_PATH", "CUDA_HOME", "CUDA_ROOT", "CUDA_TOOLKIT_ROOT_DIR"];

/// A valid CUDA root must have version.json or lib/x64 — rules out /usr, /home, etc.
fn is_cuda_root(p: &std::path::Path) -> bool {
    p.exists()
        && (p.join("version.json").exists()
            || p.join("lib").join("x64").exists()
            || p.join("include").join("cuda_runtime.h").exists())
}

fn find_cuda_path() -> PathBuf {
    // 1. Workspace .env (explicit user intent, takes priority over ambient shell env)
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let env_file = manifest
        .parent().expect("lib/cuda_rs has no parent")
        .parent().expect("lib has no parent")
        .join(".env");

    if let Ok(contents) = std::fs::read_to_string(&env_file) {
        for line in contents.lines() {
            let line = line.trim_end_matches('\r').trim();
            if line.is_empty() || line.starts_with('#') { continue; }
            if let Some((key, val)) = line.split_once('=') {
                let key = key.trim();
                let val = val.trim().trim_end_matches('\r');
                if CUDA_VARS.contains(&key) && !val.is_empty() {
                    let p = PathBuf::from(val);
                    println!("cargo:warning=cuda_rs: .env {key}={val}");
                    return p; // trust .env unconditionally
                }
            }
        }
    }

    // 2. Shell environment variables (filtered — skip non-CUDA paths like /usr)
    for var in CUDA_VARS {
        if let Ok(v) = std::env::var(var) {
            let v = v.trim().to_owned();
            if !v.is_empty() {
                let p = PathBuf::from(&v);
                if is_cuda_root(&p) {
                    println!("cargo:warning=cuda_rs: env {var}={v}");
                    return p;
                }
            }
        }
    }

    panic!(
        "CUDA toolkit not found.\n\
         Add CUDA_PATH to the workspace .env file, e.g.:\n\
         CUDA_PATH=C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.3\n\
         Install the full CUDA Toolkit from: https://developer.nvidia.com/cuda-toolkit"
    );
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=../../.env");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_ROOT");
    println!("cargo:rerun-if-env-changed=CUDA_TOOLKIT_ROOT_DIR");

    let cuda_path = find_cuda_path();
    let lib_dir = cuda_path.join("lib").join("x64");

    assert!(
        lib_dir.exists(),
        "CUDA lib/x64 not found at: {}\n\
         The toolkit at {} is present but incomplete (missing lib\\x64\\).\n\
         Reinstall using the NVIDIA CUDA Toolkit installer and select:\n\
         Custom install → CUDA Toolkit → Development → Libraries (Static + Dynamic)\n\
         https://developer.nvidia.com/cuda-toolkit",
        lib_dir.display(),
        cuda_path.display()
    );

    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=cublas");
    println!("cargo:rustc-link-lib=dylib=cublasLt");
}
