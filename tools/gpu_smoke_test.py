# app_stremlit/tools/gpu_smoke_test.py
import os
import sys
import torch

def main():
    print("=== TORCH / CUDA SMOKE TEST ===")
    print(f"torch.__version__   : {torch.__version__}")
    print(f"CUDA available      : {torch.cuda.is_available()}")
    print(f"GPU count           : {torch.cuda.device_count()}")

    if torch.cuda.is_available():
        dev = torch.device("cuda:0")
        print(f"Using device        : {torch.cuda.get_device_name(0)}")
        try:
            x = torch.randn((1024, 1024), device=dev)
            y = torch.mm(x, x.t())
            print(f"Tensor op OK, y.mean={y.mean().item():.6f}")
            print("✅ CUDA path looks good.")
            sys.exit(0)
        except Exception as e:
            print(f"❌ CUDA op failed: {e}")
            sys.exit(2)
    else:
        print("⚠️ CUDA not available. If you expected GPU, check drivers / CUDA runtime / wheel channel.")
        sys.exit(1)

if __name__ == "__main__":
    main()
