use candle_core::Device;

/// Select the best available compute device.
///
/// Tries CUDA device 0 first when compiled with the `cuda` feature. Logs the
/// outcome via `on_log` and always returns a valid device, falling back to CPU
/// if CUDA is unavailable or encounters an error.
pub fn pick_device(on_log: &impl Fn(String)) -> Device {
    #[cfg(feature = "cuda")]
    {
        match Device::cuda_if_available(0) {
            Ok(dev) if dev.is_cuda() => {
                on_log("CUDA GPU detected — using GPU".into());
                return dev;
            }
            Ok(_) => on_log("No CUDA GPU found — falling back to CPU".into()),
            Err(e) => on_log(format!("CUDA probe failed ({e}) — falling back to CPU")),
        }
    }
    #[cfg(not(feature = "cuda"))]
    on_log("CUDA not compiled in — using CPU".into());
    Device::Cpu
}
