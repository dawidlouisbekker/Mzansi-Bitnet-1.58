use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use cuda_rs::{CublasHandle, DeviceBuffer, device};
use half::{bf16, f16};

fn gpu_available() -> bool {
    device::device_count().map(|n| n > 0).unwrap_or(false)
}

// (label, m, n, k)
// Square sizes exercise peak throughput; ffn_1x4k models batch=1 inference.
const SIZES: &[(&str, usize, usize, usize)] = &[
    ("512",      512,  512,  512),
    ("1024",    1024, 1024, 1024),
    ("2048",    2048, 2048, 2048),
    ("ffn_1x4k",   1, 4096, 4096),
];

fn bench_matmul_f32(c: &mut Criterion) {
    if !gpu_available() { return; }
    let handle = CublasHandle::new().unwrap();
    let mut group = c.benchmark_group("matmul_f32");

    for &(label, m, n, k) in SIZES {
        group.throughput(Throughput::Elements(2 * m as u64 * n as u64 * k as u64));

        let a = DeviceBuffer::from_slice(&vec![1.0_f32; m * k]).unwrap();
        let b = DeviceBuffer::from_slice(&vec![1.0_f32; k * n]).unwrap();
        let mut c_buf = DeviceBuffer::<f32>::uninit(m * n).unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(label), &(m, n, k),
            |bencher, &(m, n, k)| {
                bencher.iter(|| {
                    handle.matmul_f32(m, n, k, &a, &b, &mut c_buf).unwrap();
                    device::synchronize().unwrap();
                });
            },
        );
    }
    group.finish();
}

fn bench_matmul_f16(c: &mut Criterion) {
    if !gpu_available() { return; }
    let handle = CublasHandle::new().unwrap();
    let mut group = c.benchmark_group("matmul_f16");

    for &(label, m, n, k) in SIZES {
        group.throughput(Throughput::Elements(2 * m as u64 * n as u64 * k as u64));

        let a = DeviceBuffer::from_slice(&vec![f16::from_f32(1.0); m * k]).unwrap();
        let b = DeviceBuffer::from_slice(&vec![f16::from_f32(1.0); k * n]).unwrap();
        let mut c_buf = DeviceBuffer::<f16>::uninit(m * n).unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(label), &(m, n, k),
            |bencher, &(m, n, k)| {
                bencher.iter(|| {
                    handle.matmul_f16(m, n, k, &a, &b, &mut c_buf).unwrap();
                    device::synchronize().unwrap();
                });
            },
        );
    }
    group.finish();
}

fn bench_matmul_bf16(c: &mut Criterion) {
    if !gpu_available() { return; }
    let handle = CublasHandle::new().unwrap();
    let mut group = c.benchmark_group("matmul_bf16");

    for &(label, m, n, k) in SIZES {
        group.throughput(Throughput::Elements(2 * m as u64 * n as u64 * k as u64));

        let a = DeviceBuffer::from_slice(&vec![bf16::from_f32(1.0); m * k]).unwrap();
        let b = DeviceBuffer::from_slice(&vec![bf16::from_f32(1.0); k * n]).unwrap();
        let mut c_buf = DeviceBuffer::<bf16>::uninit(m * n).unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(label), &(m, n, k),
            |bencher, &(m, n, k)| {
                bencher.iter(|| {
                    handle.matmul_bf16(m, n, k, &a, &b, &mut c_buf).unwrap();
                    device::synchronize().unwrap();
                });
            },
        );
    }
    group.finish();
}

fn bench_matmul_i8_i32(c: &mut Criterion) {
    if !gpu_available() { return; }
    let handle = CublasHandle::new().unwrap();
    let mut group = c.benchmark_group("matmul_i8_i32");

    for &(label, m, n, k) in SIZES {
        group.throughput(Throughput::Elements(2 * m as u64 * n as u64 * k as u64));

        let a = DeviceBuffer::from_slice(&vec![1_i8; m * k]).unwrap();
        let b = DeviceBuffer::from_slice(&vec![1_i8; k * n]).unwrap();
        let mut c_buf = DeviceBuffer::<i32>::uninit(m * n).unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(label), &(m, n, k),
            |bencher, &(m, n, k)| {
                bencher.iter(|| {
                    handle.matmul_i8_i32(m, n, k, &a, &b, &mut c_buf).unwrap();
                    device::synchronize().unwrap();
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_matmul_f32, bench_matmul_f16, bench_matmul_bf16, bench_matmul_i8_i32);
criterion_main!(benches);
