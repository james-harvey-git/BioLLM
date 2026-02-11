from __future__ import annotations

import json

import torch


def main() -> int:
    info = {
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "torch_version": torch.__version__,
    }

    if torch.cuda.is_available():
        dev = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(dev)
        info.update(
            {
                "cuda_current_device": int(dev),
                "cuda_device_name": torch.cuda.get_device_name(dev),
                "cuda_capability": f"{props.major}.{props.minor}",
                "cuda_total_mem_gb": round(props.total_memory / (1024 ** 3), 2),
            }
        )

    print(json.dumps(info, indent=2))

    if not torch.cuda.is_available():
        print("\nCUDA is not available. On your gaming PC verify NVIDIA driver + CUDA PyTorch install.")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
