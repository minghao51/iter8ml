"""Hardware profile auto-detection for model routing decisions."""

import os
import platform

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

    @classmethod
    def _get_default_threads(cls) -> int:
        """Get default thread count based on platform."""
        if platform.system() == "Darwin" and platform.machine() == "arm64":
            return 1  # macOS ARM64 has performance issues with threading
        return os.cpu_count() or 1

    @classmethod
    def configure_omp_threads(cls, threads: int | None = None) -> int:
        """Configure OMP_NUM_THREADS environment variable.

        Args:
            threads: Number of threads to use. If None, uses platform default.

        Returns:
            The configured thread count.
        """
        thread_count = threads or cls._get_default_threads()
        if threads is None:
            # Only set if not already configured by user
            os.environ.setdefault("OMP_NUM_THREADS", str(thread_count))
        else:
            # Explicitly specified, so override
            os.environ["OMP_NUM_THREADS"] = str(thread_count)
        return thread_count
