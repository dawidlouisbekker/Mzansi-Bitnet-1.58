use cuda_rs::{CublasHandle, DeviceBuffer, device};
use half::{bf16, f16};

// Returns early from the enclosing test if no CUDA device is found.
macro_rules! require_gpu {
    () => {
        match cuda_rs::device::device_count() {
            Ok(n) if n > 0 => {}
            _ => {
                eprintln!("SKIP: no CUDA device found");
                return;
            }
        }
    };
}

// A(M×K) all-ones · B(K×N) all-ones = C(M×N) all-K.
// K=4, so every output element should equal 4.

#[test]
fn test_matmul_f32_correctness() {
    require_gpu!();
    let handle = CublasHandle::new().unwrap();
    let (m, n, k) = (4, 4, 4);

    let a = DeviceBuffer::from_slice(&[1.0_f32; 16]).unwrap();
    let b = DeviceBuffer::from_slice(&[1.0_f32; 16]).unwrap();
    let mut c = DeviceBuffer::<f32>::uninit(16).unwrap();

    handle.matmul_f32(m, n, k, &a, &b, &mut c).unwrap();
    device::synchronize().unwrap();

    for (i, val) in c.to_vec().unwrap().iter().enumerate() {
        assert!((val - 4.0_f32).abs() < 1e-4, "elem {i}: expected 4.0, got {val}");
    }
}

#[test]
fn test_matmul_f16_correctness() {
    require_gpu!();
    let handle = CublasHandle::new().unwrap();
    let (m, n, k) = (4, 4, 4);

    let a = DeviceBuffer::from_slice(&[f16::from_f32(1.0); 16]).unwrap();
    let b = DeviceBuffer::from_slice(&[f16::from_f32(1.0); 16]).unwrap();
    let mut c = DeviceBuffer::<f16>::uninit(16).unwrap();

    handle.matmul_f16(m, n, k, &a, &b, &mut c).unwrap();
    device::synchronize().unwrap();

    for (i, val) in c.to_vec().unwrap().iter().enumerate() {
        let v = val.to_f32();
        assert!((v - 4.0_f32).abs() < 0.1, "elem {i}: expected ~4.0, got {v}");
    }
}

#[test]
fn test_matmul_bf16_correctness() {
    require_gpu!();
    let handle = CublasHandle::new().unwrap();
    let (m, n, k) = (4, 4, 4);

    let a = DeviceBuffer::from_slice(&[bf16::from_f32(1.0); 16]).unwrap();
    let b = DeviceBuffer::from_slice(&[bf16::from_f32(1.0); 16]).unwrap();
    let mut c = DeviceBuffer::<bf16>::uninit(16).unwrap();

    handle.matmul_bf16(m, n, k, &a, &b, &mut c).unwrap();
    device::synchronize().unwrap();

    for (i, val) in c.to_vec().unwrap().iter().enumerate() {
        let v = val.to_f32();
        assert!((v - 4.0_f32).abs() < 0.1, "elem {i}: expected ~4.0, got {v}");
    }
}

#[test]
fn test_matmul_i8_i32_correctness() {
    require_gpu!();
    let handle = CublasHandle::new().unwrap();
    let (m, n, k) = (4, 4, 4);

    let a = DeviceBuffer::from_slice(&[1_i8; 16]).unwrap();
    let b = DeviceBuffer::from_slice(&[1_i8; 16]).unwrap();
    let mut c = DeviceBuffer::<i32>::uninit(16).unwrap();

    handle.matmul_i8_i32(m, n, k, &a, &b, &mut c).unwrap();
    device::synchronize().unwrap();

    for (i, val) in c.to_vec().unwrap().iter().enumerate() {
        assert_eq!(*val, 4_i32, "elem {i}: expected 4, got {val}");
    }
}

// Non-square: 2×4 · 4×3 = 2×3, every element = 4.
// Guards the A/B swap + dimension flip in the row-major trick inside cublas.rs —
// a bug there is invisible with square matrices but shows up here.
#[test]
fn test_matmul_f32_non_square() {
    require_gpu!();
    let handle = CublasHandle::new().unwrap();

    let a = DeviceBuffer::from_slice(&[1.0_f32; 2 * 4]).unwrap();
    let b = DeviceBuffer::from_slice(&[1.0_f32; 4 * 3]).unwrap();
    let mut c = DeviceBuffer::<f32>::uninit(2 * 3).unwrap();

    handle.matmul_f32(2, 3, 4, &a, &b, &mut c).unwrap();
    device::synchronize().unwrap();

    let result = c.to_vec().unwrap();
    assert_eq!(result.len(), 6);
    for (i, val) in result.iter().enumerate() {
        assert!((val - 4.0_f32).abs() < 1e-4, "elem {i}: expected 4.0, got {val}");
    }
}
