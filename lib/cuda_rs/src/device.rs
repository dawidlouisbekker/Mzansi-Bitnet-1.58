use crate::error::{Result, cuda_check};
use crate::ffi::rt;
use std::ffi::c_int;

/// Returns the number of CUDA-capable devices on the system.
pub fn device_count() -> Result<usize> {
    let mut count: c_int = 0;
    unsafe { cuda_check(rt::cudaGetDeviceCount(&raw mut count))? };
    Ok(count as usize)
}

/// Selects the CUDA device used by the calling thread (0-indexed).
pub fn set_device(index: usize) -> Result<()> {
    unsafe { cuda_check(rt::cudaSetDevice(index as c_int)) }
}

/// Blocks the calling thread until all previously issued CUDA commands finish.
pub fn synchronize() -> Result<()> {
    unsafe { cuda_check(rt::cudaDeviceSynchronize()) }
}
