import subprocess


def _nvidia_smi_vram():
    """Query VRAM via nvidia-smi; returns list of (name, vram_mb) tuples."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        results = []
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 2:
                name, vram_mb = parts[0], int(parts[1])
                results.append((name, vram_mb))
        return results
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError):
        return []


def _torch_vram():
    """Query VRAM via PyTorch CUDA; returns list of (name, vram_mb) tuples."""
    try:
        import torch

        if not torch.cuda.is_available():
            return []
        results = []
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            vram_bytes = torch.cuda.get_device_properties(i).total_memory
            results.append((name, vram_bytes // (1024 ** 2)))
        return results
    except ImportError:
        return []


def scan_gpus():
    gpus = _torch_vram() or _nvidia_smi_vram()
    return gpus


def main():
    gpus = scan_gpus()
    if not gpus:
        print("No GPU detected.")
        return

    print(f"{'#':<4} {'GPU Name':<50} {'VRAM':>10}")
    print("-" * 66)
    for i, (name, vram_mb) in enumerate(gpus):
        if vram_mb >= 1024:
            vram_str = f"{vram_mb / 1024:.2f} GB"
        else:
            vram_str = f"{vram_mb} MB"
        print(f"{i:<4} {name:<50} {vram_str:>10}")


if __name__ == "__main__":
    main()
