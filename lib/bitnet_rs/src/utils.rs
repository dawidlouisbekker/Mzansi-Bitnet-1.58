use candle_core::Device;

/// Returns `Device::Cpu`. All candle tensors run on CPU; GPU acceleration for
/// matrix multiplication is handled transparently by cuda_rs (cuBLAS) inside
/// `bitlinear_forward` when the `cuda` feature is enabled.
pub fn pick_device(on_log: &impl Fn(String)) -> Device {
    #[cfg(feature = "cuda_rs")]
    {
        use cuda_rs::CublasHandle;
        match CublasHandle::new() {
            Ok(_) => on_log("cuBLAS available — BitLinear matmuls will run on GPU".into()),
            Err(e) => on_log(format!("cuBLAS init failed ({e}) — running fully on CPU")),
        }
    }
    #[cfg(not(feature = "cuda_rs"))]
    on_log("cuda feature disabled — running on CPU".into());

    Device::Cpu
}
