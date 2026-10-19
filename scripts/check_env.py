import sys
import torch
import ultralytics


def main():
    print("=" * 50)
    print("Python 环境检查")
    print("=" * 50)

    print(f"Python executable : {sys.executable}")
    print(f"Python version    : {sys.version}")
    print(f"PyTorch version   : {torch.__version__}")
    print(f"Ultralytics       : {ultralytics.__version__}")

    print("\n" + "=" * 50)
    print("GPU 环境检查")
    print("=" * 50)

    print(f"CUDA available    : {torch.cuda.is_available()}")
    print(f"PyTorch CUDA      : {torch.version.cuda}")

    if torch.cuda.is_available():
        print(f"GPU count         : {torch.cuda.device_count()}")
        print(f"GPU name          : {torch.cuda.get_device_name(0)}")

        total_memory = torch.cuda.get_device_properties(0).total_memory
        total_memory_gb = total_memory / 1024 ** 3

        print(f"GPU memory        : {total_memory_gb:.2f} GB")


if __name__ == "__main__":
    main()