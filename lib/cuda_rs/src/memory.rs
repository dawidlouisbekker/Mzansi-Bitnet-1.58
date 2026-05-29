use crate::error::{Result, cuda_check};
use crate::ffi::rt::{self, MemcpyKind};
use std::ffi::c_void;
use std::marker::PhantomData;

/// Owned GPU memory buffer for elements of type `T`.
///
/// Dropped via `cudaFree`; does not copy data back automatically.
pub struct DeviceBuffer<T> {
    ptr: *mut T,
    len: usize,
    _marker: PhantomData<T>,
}

unsafe impl<T: Send> Send for DeviceBuffer<T> {}
unsafe impl<T: Sync> Sync for DeviceBuffer<T> {}

impl<T> DeviceBuffer<T> {
    /// Allocates uninitialised GPU memory for `len` elements.
    pub fn uninit(len: usize) -> Result<Self> {
        let bytes = len * std::mem::size_of::<T>();
        let mut ptr: *mut c_void = std::ptr::null_mut();
        unsafe { cuda_check(rt::cudaMalloc(&raw mut ptr, bytes))? };
        Ok(Self { ptr: ptr as *mut T, len, _marker: PhantomData })
    }

    /// Allocates GPU memory and copies `src` from the host.
    pub fn from_slice(src: &[T]) -> Result<Self> {
        let buf = Self::uninit(src.len())?;
        let bytes = src.len() * std::mem::size_of::<T>();
        unsafe {
            cuda_check(rt::cudaMemcpy(
                buf.ptr as *mut c_void,
                src.as_ptr() as *const c_void,
                bytes,
                MemcpyKind::HostToDevice,
            ))?;
        }
        Ok(buf)
    }

    /// Copies the buffer back to a host `Vec`.
    pub fn to_vec(&self) -> Result<Vec<T>>
    where
        T: Default + Clone,
    {
        let mut out = vec![T::default(); self.len];
        let bytes = self.len * std::mem::size_of::<T>();
        unsafe {
            cuda_check(rt::cudaMemcpy(
                out.as_mut_ptr() as *mut c_void,
                self.ptr as *const c_void,
                bytes,
                MemcpyKind::DeviceToHost,
            ))?;
        }
        Ok(out)
    }

    /// Overwrites the buffer contents from a host slice (must be same length).
    pub fn copy_from_slice(&mut self, src: &[T]) -> Result<()> {
        assert_eq!(src.len(), self.len, "slice length mismatch");
        let bytes = self.len * std::mem::size_of::<T>();
        unsafe {
            cuda_check(rt::cudaMemcpy(
                self.ptr as *mut c_void,
                src.as_ptr() as *const c_void,
                bytes,
                MemcpyKind::HostToDevice,
            ))
        }
    }

    pub fn len(&self) -> usize { self.len }
    pub fn is_empty(&self) -> bool { self.len == 0 }

    pub fn as_ptr(&self) -> *const T { self.ptr }
    pub fn as_mut_ptr(&mut self) -> *mut T { self.ptr }
}

impl<T> Drop for DeviceBuffer<T> {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { rt::cudaFree(self.ptr as *mut c_void) };
        }
    }
}
