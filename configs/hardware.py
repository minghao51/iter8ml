"""Hardware profile auto-detection for model routing decisions."""

import psutil
from pydantic import BaseModel


class HardwareProfile(BaseModel):
    vram_gb: float
    system_ram_gb: float
    cpu_cores: int
    has_gpu: bool
    gpu_name: str | None

    @classmethod
    def detect(cls) -> "HardwareProfile":
        try:
            import torch

            if torch.cuda.is_available():
                vram = torch.cuda.get_device_properties(0).total_memory / 1e9
                has_gpu = True
                gpu_name = torch.cuda.get_device_name(0)
            else:
                vram = 0.0
                has_gpu = False
                gpu_name = None
        except ImportError:
            vram = 0.0
            has_gpu = False
            gpu_name = None

        return cls(
            vram_gb=round(vram, 1),
            system_ram_gb=round(psutil.virtual_memory().total / 1e9, 1),
            cpu_cores=psutil.cpu_count(logical=False),
            has_gpu=has_gpu,
            gpu_name=gpu_name,
        )
