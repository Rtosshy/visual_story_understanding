import torch
import subprocess
import re

def get_cuda_runtime_version():
    # Prefer torch reported compile/runtime version
    if torch.version.cuda:
        return torch.version.cuda
    # Fallback: parse nvidia-smi
    try:
        out = subprocess.check_output(["nvidia-smi"], text=True)
        m = re.search(r"CUDA Version:\s*([\d.]+)", out)
        if m:
            return m.group(1)
    except Exception:
        pass
    # Fallback: parse nvcc --version
    try:
        out = subprocess.check_output(["nvcc", "--version"], text=True)
        m = re.search(r"release\s+(\d+\.\d+)", out)
        if m:
            return m.group(1)
    except Exception:
        pass
    return "Unknown"

def main():
    torch_version = torch.__version__
    cuda_available = torch.cuda.is_available()
    cuda_runtime = get_cuda_runtime_version()
    num_devices = torch.cuda.device_count() if cuda_available else 0
    if cuda_available:
        current_idx = torch.cuda.current_device()
        current_name = torch.cuda.get_device_name(current_idx)
    else:
        current_idx = None
        current_name = None

    print(f"PyTorch version: {torch_version}")
    print(f"CUDA available: {cuda_available}")
    print(f"CUDA runtime version: {cuda_runtime}")
    print(f"Number of CUDA devices: {num_devices}")
    if cuda_available:
        print(f"Current CUDA device index: {current_idx}")
        print(f"Current CUDA device name: {current_name}")

    # Example: pick device for later use
    device = torch.device("cuda" if cuda_available else "cpu")
    # print(f"Using device object: {device}")

if __name__ == "__main__":
    main()