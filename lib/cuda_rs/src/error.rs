use std::ffi::CStr;

#[derive(Debug, thiserror::Error)]
pub enum CudaError {
    #[error("CUDA runtime error {code}: {message}")]
    Runtime { code: i32, message: String },
    #[error("cuBLAS error {0}")]
    Cublas(i32),
    #[error("invalid argument: {0}")]
    InvalidArg(&'static str),
}

pub type Result<T> = std::result::Result<T, CudaError>;

pub(crate) fn cuda_check(code: i32) -> Result<()> {
    if code == 0 {
        return Ok(());
    }
    let message = unsafe {
        let ptr = crate::ffi::rt::cudaGetErrorString(code);
        if ptr.is_null() {
            "unknown error".to_owned()
        } else {
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    };
    Err(CudaError::Runtime { code, message })
}

pub(crate) fn cublas_check(code: i32) -> Result<()> {
    if code == 0 {
        Ok(())
    } else {
        Err(CudaError::Cublas(code))
    }
}
