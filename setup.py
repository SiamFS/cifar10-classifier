import subprocess
import sys


def detect_gpu():
    try:
        output = subprocess.check_output(
            "nvidia-smi --query-gpu=name,memory.total,driver_version,cuda_version --format=csv,noheader",
            shell=True,
            stderr=subprocess.DEVNULL,
            encoding="utf-8"
        ).strip()
        if output:
            return output
    except subprocess.CalledProcessError:
        pass
    return None


def main():
    print("=" * 50)
    print("  CIFAR-10 GPU Detection & Setup")
    print("=" * 50)
    print()

    gpu_info = detect_gpu()

    if gpu_info:
        print("[GPU DETECTED]")
        parts = gpu_info.split(", ")
        name = parts[0].strip()
        memory = parts[1].strip()
        driver = parts[2].strip()
        cuda_ver = parts[3].strip()
        print(f"  Name:        {name}")
        print(f"  VRAM:        {memory}")
        print(f"  Driver:      {driver}")
        print(f"  CUDA:        {cuda_ver}")
        print()
        print("[INSTALLING] PyTorch with CUDA support...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "torch", "torchvision", "torchaudio",
            "--index-url", "https://download.pytorch.org/whl/cu124"
        ])
        print()
        print("[GPU OPTIMIZATIONS ENABLED]")
        print("  - torch.cuda.amp (Automatic Mixed Precision)")
        print("  - torch.backends.cudnn.benchmark = True")
        print("  - Pin memory for DataLoader")
        print("  - channels_last memory format")
        print()
        print("[SUCCESS] Setup complete! Run: python train.py")
    else:
        print("[WARNING] No NVIDIA GPU detected!")
        print("  Training will be SLOW on CPU.")
        print("  CIFAR-10 with 200 epochs may take HOURS instead of minutes.")
        print()
        response = input("  Continue with CPU-only PyTorch? (y/n): ").strip().lower()
        if response != "y":
            print("[ABORTED] Install NVIDIA GPU drivers and try again.")
            sys.exit(1)
        print()
        print("[INSTALLING] PyTorch CPU-only...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "torch", "torchvision", "torchaudio",
            "--index-url", "https://download.pytorch.org/whl/cpu"
        ])
        print()
        print("[WARNING] Running on CPU. Expect slow training times.")
        print("  Set --device cpu in scripts or environment variable CUDA_VISIBLE_DEVICES=-1")

    print()
    print("[INSTALLING] Other dependencies...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "numpy", "matplotlib", "seaborn", "scikit-learn",
        "gradio", "tqdm", "pillow", "pandas",
    ])
    print()
    print("=" * 50)
    print("  Setup complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()
