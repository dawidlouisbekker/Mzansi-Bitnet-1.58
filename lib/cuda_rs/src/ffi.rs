/// Raw FFI bindings — never expose these outside this crate.

pub mod rt {
    use std::ffi::{c_int, c_void, c_char};

    #[repr(i32)]
    #[allow(dead_code)]
    pub enum MemcpyKind {
        HostToHost     = 0,
        HostToDevice   = 1,
        DeviceToHost   = 2,
        DeviceToDevice = 3,
    }

    unsafe extern "C" {
        pub fn cudaMalloc(dev_ptr: *mut *mut c_void, size: usize) -> c_int;
        pub fn cudaFree(dev_ptr: *mut c_void) -> c_int;
        pub fn cudaMemcpy(
            dst: *mut c_void,
            src: *const c_void,
            count: usize,
            kind: MemcpyKind,
        ) -> c_int;
        pub fn cudaDeviceSynchronize() -> c_int;
        pub fn cudaGetErrorString(error: c_int) -> *const c_char;
        pub fn cudaGetDeviceCount(count: *mut c_int) -> c_int;
        pub fn cudaSetDevice(device: c_int) -> c_int;
    }
}

pub mod blas {
    use std::ffi::{c_int, c_void};

    pub type Handle = *mut c_void;

    /// Column-major operation selector.
    #[repr(i32)]
    #[derive(Clone, Copy)]
    pub enum Op {
        N = 0, // no-transpose
        T = 1, // transpose
        //C = 2, // conjugate transpose
    }

    /// CUDA scalar / tensor element types used by GemmEx.
    #[repr(u32)]
    #[derive(Clone, Copy)]
    #[allow(dead_code)]
    pub enum DataType {
        R32F  = 0,  // f32
        R16F  = 2,  // f16
        R8I   = 3,  // i8
        R32I  = 10, // i32
        R16BF = 14, // bf16
    }

    /// cuBLAS compute type for GemmEx.
    #[repr(u32)]
    #[derive(Clone, Copy)]
    #[allow(dead_code)]
    pub enum ComputeType {
        F32     = 68, // standard f32 accumulation
        F32Fast = 74, // TF32 (tensor-core) accumulation
        I32     = 70, // i32 accumulation (for INT8 inputs)
    }

    // CUBLAS_GEMM_DEFAULT — let cuBLAS choose the best algorithm.
    pub const GEMM_DEFAULT: c_int = -1;

    unsafe extern "C" {
        pub fn cublasCreate_v2(handle: *mut Handle) -> c_int;
        pub fn cublasDestroy_v2(handle: Handle) -> c_int;

        /// F32 GEMM:  C = alpha * op(A) * op(B) + beta * C
        /// All matrices are column-major.
        pub fn cublasSgemm_v2(
            handle: Handle,
            transa: Op,
            transb: Op,
            m: c_int, n: c_int, k: c_int,
            alpha: *const f32,
            A: *const f32, lda: c_int,
            B: *const f32, ldb: c_int,
            beta: *const f32,
            C: *mut f32, ldc: c_int,
        ) -> c_int;

        /// F16 GEMM  (half-precision inputs and output).
        pub fn cublasHgemm(
            handle: Handle,
            transa: Op,
            transb: Op,
            m: c_int, n: c_int, k: c_int,
            alpha: *const half::f16,
            A: *const half::f16, lda: c_int,
            B: *const half::f16, ldb: c_int,
            beta: *const half::f16,
            C: *mut half::f16, ldc: c_int,
        ) -> c_int;

        /// Mixed-precision GEMM — supports INT8/BF16/F16/F32 in any combination.
        pub fn cublasGemmEx(
            handle: Handle,
            transa: Op,
            transb: Op,
            m: c_int, n: c_int, k: c_int,
            alpha: *const c_void,
            A: *const c_void, a_type: DataType, lda: c_int,
            B: *const c_void, b_type: DataType, ldb: c_int,
            beta: *const c_void,
            C: *mut c_void,   c_type: DataType, ldc: c_int,
            compute_type: ComputeType,
            algo: c_int,
        ) -> c_int;
    }
}
