use crate::error::{Result, cublas_check};
use crate::ffi::blas::{self, ComputeType, DataType, Handle, Op, GEMM_DEFAULT};
use crate::memory::DeviceBuffer;
use std::ffi::c_void;

/// Wraps a `cublasHandle_t`; destroyed on drop.
pub struct CublasHandle(Handle);

// cuBLAS handles are safe to share across threads.
unsafe impl Send for CublasHandle {}
unsafe impl Sync for CublasHandle {}

impl CublasHandle {
    pub fn new() -> Result<Self> {
        let mut h: Handle = std::ptr::null_mut();
        unsafe { cublas_check(blas::cublasCreate_v2(&raw mut h))? };
        Ok(Self(h))
    }

    // ── F32 GEMM ─────────────────────────────────────────────────────────────

    /// Column-major F32 GEMM:  `C = alpha * op(A) * op(B) + beta * C`
    ///
    /// - `A`: (m × k) or (k × m) depending on `trans_a`
    /// - `B`: (k × n) or (n × k) depending on `trans_b`
    /// - `C`: (m × n)  — in/out
    #[allow(clippy::too_many_arguments)]
    pub fn sgemm(
        &self,
        trans_a: bool, trans_b: bool,
        m: i32, n: i32, k: i32,
        alpha: f32,
        a: &DeviceBuffer<f32>, lda: i32,
        b: &DeviceBuffer<f32>, ldb: i32,
        beta: f32,
        c: &mut DeviceBuffer<f32>, ldc: i32,
    ) -> Result<()> {
        let op_a = if trans_a { Op::T } else { Op::N };
        let op_b = if trans_b { Op::T } else { Op::N };
        unsafe {
            cublas_check(blas::cublasSgemm_v2(
                self.0, op_a, op_b,
                m, n, k,
                &alpha, a.as_ptr(), lda,
                        b.as_ptr(), ldb,
                &beta,  c.as_mut_ptr(), ldc,
            ))
        }
    }

    /// Row-major F32 matmul convenience wrapper:  `C (M×N) = A (M×K) * B (K×N)`
    ///
    /// Internally swaps A/B and their leading dimensions so cuBLAS (column-major)
    /// produces the correct row-major result without any data transposition.
    pub fn matmul_f32(
        &self,
        m: usize, n: usize, k: usize,
        a: &DeviceBuffer<f32>,
        b: &DeviceBuffer<f32>,
        c: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        // cuBLAS column-major trick: C^T = B^T * A^T
        // Swap A↔B, swap m↔n, leading dims follow the original row-major strides.
        unsafe {
            cublas_check(blas::cublasSgemm_v2(
                self.0,
                Op::N, Op::N,
                n as i32, m as i32, k as i32,
                &1.0_f32,
                b.as_ptr(), n as i32,
                a.as_ptr(), k as i32,
                &0.0_f32,
                c.as_mut_ptr(), n as i32,
            ))
        }
    }

    // ── F16 GEMM ─────────────────────────────────────────────────────────────

    /// Row-major F16 matmul:  `C (M×N) = A (M×K) * B (K×N)`
    pub fn matmul_f16(
        &self,
        m: usize, n: usize, k: usize,
        a: &DeviceBuffer<half::f16>,
        b: &DeviceBuffer<half::f16>,
        c: &mut DeviceBuffer<half::f16>,
    ) -> Result<()> {
        let alpha = half::f16::from_f32(1.0);
        let beta  = half::f16::from_f32(0.0);
        unsafe {
            cublas_check(blas::cublasHgemm(
                self.0,
                Op::N, Op::N,
                n as i32, m as i32, k as i32,
                &alpha,
                b.as_ptr(), n as i32,
                a.as_ptr(), k as i32,
                &beta,
                c.as_mut_ptr(), n as i32,
            ))
        }
    }

    // ── INT8 GEMM (BitNet-style quantised matmul) ─────────────────────────────

    /// Row-major INT8 matmul with I32 accumulation.
    ///
    /// Typical BitNet use-case: ternary weights packed as `i8`, activations
    /// quantised to `i8`, result accumulated in `i32` for rescaling.
    ///
    /// - `a`: activations  (M×K), i8
    /// - `b`: weight matrix (K×N), i8
    /// - `c`: output        (M×N), i32
    pub fn matmul_i8_i32(
        &self,
        m: usize, n: usize, k: usize,
        a: &DeviceBuffer<i8>,
        b: &DeviceBuffer<i8>,
        c: &mut DeviceBuffer<i32>,
    ) -> Result<()> {
        let alpha: i32 = 1;
        let beta:  i32 = 0;
        unsafe {
            cublas_check(blas::cublasGemmEx(
                self.0,
                Op::N, Op::N,
                n as i32, m as i32, k as i32,
                &alpha as *const i32 as *const c_void,
                b.as_ptr() as *const c_void, DataType::R8I, n as i32,
                a.as_ptr() as *const c_void, DataType::R8I, k as i32,
                &beta  as *const i32 as *const c_void,
                c.as_mut_ptr() as *mut c_void, DataType::R32I, n as i32,
                ComputeType::I32,
                GEMM_DEFAULT,
            ))
        }
    }

    // ── BF16 GEMM ────────────────────────────────────────────────────────────

    /// Row-major BF16 matmul with F32 TF32 accumulation.
    ///
    /// Uses `GemmEx` because there is no dedicated `cublasB16gemm` entry point.
    pub fn matmul_bf16(
        &self,
        m: usize, n: usize, k: usize,
        a: &DeviceBuffer<half::bf16>,
        b: &DeviceBuffer<half::bf16>,
        c: &mut DeviceBuffer<half::bf16>,
    ) -> Result<()> {
        let alpha = half::bf16::from_f32(1.0);
        let beta  = half::bf16::from_f32(0.0);
        unsafe {
            cublas_check(blas::cublasGemmEx(
                self.0,
                Op::N, Op::N,
                n as i32, m as i32, k as i32,
                &alpha as *const half::bf16 as *const c_void,
                b.as_ptr() as *const c_void, DataType::R16BF, n as i32,
                a.as_ptr() as *const c_void, DataType::R16BF, k as i32,
                &beta  as *const half::bf16 as *const c_void,
                c.as_mut_ptr() as *mut c_void, DataType::R16BF, n as i32,
                ComputeType::F32Fast,
                GEMM_DEFAULT,
            ))
        }
    }
}

impl Drop for CublasHandle {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe { blas::cublasDestroy_v2(self.0) };
        }
    }
}
