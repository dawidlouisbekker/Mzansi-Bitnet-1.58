use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use memmap2::MmapOptions;
use safetensors::SafeTensors;

/// Holds the mmap handle so the OS mapping stays alive as long as the tensors are in use.
pub struct MmapWeights {
    // The Mmap must outlive all Tensors created from it.
    // Tensors are copied into candle-managed storage; the mmap is kept
    // so the OS streams pages on first access rather than reading into a Vec first.
    _mmap: memmap2::Mmap,
    pub tensors: HashMap<String, Tensor>,
}

/// Open a safetensors file via memory-map and parse every tensor into a candle Tensor.
///
/// The OS loads pages on demand (not upfront), so startup is fast even for 4 GB models.
/// All tensors are loaded onto `device` in their stored dtype (bf16 for BitNet weights).
pub fn load_mmap(path: &Path, device: &Device) -> Result<MmapWeights> {
    let file = std::fs::File::open(path)
        .with_context(|| format!("cannot open {}", path.display()))?;

    let mmap = unsafe {
        MmapOptions::new()
            .map(&file)
            .with_context(|| format!("mmap failed for {}", path.display()))?
    };

    let st = SafeTensors::deserialize(&mmap)
        .with_context(|| "safetensors parse error")?;

    let mut tensors = HashMap::with_capacity(st.len());
    for (name, view) in st.tensors() {
        let shape: Vec<usize> = view.shape().to_vec();
        let dtype = st_dtype_to_candle(view.dtype())?;
        let tensor = Tensor::from_raw_buffer(view.data(), dtype, &shape, device)
            .with_context(|| format!("tensor {name} failed to load"))?;
        tensors.insert(name.to_string(), tensor);
    }

    Ok(MmapWeights { _mmap: mmap, tensors })
}

fn st_dtype_to_candle(dtype: safetensors::Dtype) -> Result<DType> {
    match dtype {
        safetensors::Dtype::F32  => Ok(DType::F32),
        safetensors::Dtype::F16  => Ok(DType::F16),
        safetensors::Dtype::BF16 => Ok(DType::BF16),
        safetensors::Dtype::I32  => Ok(DType::I64),
        safetensors::Dtype::I64  => Ok(DType::I64),
        other => anyhow::bail!("unsupported safetensors dtype: {other:?}"),
    }
}
