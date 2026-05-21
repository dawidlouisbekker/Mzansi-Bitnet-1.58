use anyhow::Result;
use candle_core::{DType, Device, Tensor, Var};

/// A linear layer whose output is `base(x) + scale * (x @ A^T @ B^T)`.
///
/// Only `a` and `b` hold gradients; the frozen `base` weight never accumulates one.
pub struct LoraLinear {
    base: Tensor,
    pub a: Var,
    pub b: Var,
    scale: f64,
}

impl LoraLinear {
    /// Create a LoRA adapter over a frozen weight matrix.
    ///
    /// `base`  — frozen bf16 weight, shape `[out, in]`
    /// `rank`  — LoRA rank (r)
    /// `alpha` — LoRA alpha; effective scale = alpha / rank
    pub fn new(base: Tensor, rank: usize, alpha: f64, device: &Device) -> Result<Self> {
        let (out_f, in_f) = base.dims2()?;
        let scale = alpha / rank as f64;

        // A ~ Kaiming uniform: U(-1/√in, 1/√in)
        let bound = (1.0_f64 / in_f as f64).sqrt();
        let a_data = Tensor::rand(-bound, bound, (rank, in_f), device)?.to_dtype(DType::F32)?;
        let a = Var::from_tensor(&a_data)?;

        // B initialised to zero so LoRA output starts at zero
        let b_data = Tensor::zeros((out_f, rank), DType::F32, device)?;
        let b = Var::from_tensor(&b_data)?;

        Ok(Self { base, a, b, scale })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_shape = x.dims().to_vec();
        let in_dim = *x_shape.last().unwrap();
        let leading: usize = x_shape[..x_shape.len() - 1].iter().product();

        let flat = x.reshape((leading, in_dim))?;

        let flat_f32 = flat.to_dtype(DType::F32)?;
        let base_out = flat_f32.matmul(&self.base.to_dtype(DType::F32)?.t()?)?;
        let lora_out = flat_f32
            .matmul(&self.a.as_tensor().t()?)?
            .matmul(&self.b.as_tensor().t()?)?
            .affine(self.scale, 0.0)?;

        let combined = (base_out + lora_out)?;
        let out_features = combined.dim(1)?;
        let mut out_shape = x_shape;
        *out_shape.last_mut().unwrap() = out_features;
        Ok(combined.reshape(out_shape)?)
    }

    /// Return the trainable variables so the optimizer can iterate over them.
    pub fn vars(&self) -> [&Var; 2] {
        [&self.a, &self.b]
    }
}
