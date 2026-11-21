"""Wykrywanie GPU i środowiska. Działa przez NVML, a gdy go brak - nvidia-smi / torch."""
from __future__ import annotations

import platform
import re
import socket
import subprocess
from dataclasses import asdict, dataclass, field


@dataclass
class GPUInfo:
    index: int = 0
    name: str = "CPU"
    memory_total_gb: float = 0.0
    compute_capability: str = ""
    driver: str = ""
    cuda_runtime: str = ""
    count: int = 0
    supports_bf16: bool = False
    power_limit_w: float | None = None
    extra: dict = field(default_factory=dict)

    @property
    def slug(self) -> str:
        """Nazwa bezpieczna do użycia w ścieżkach, np. 'nvidia-a100-sxm4-40gb'."""
        s = re.sub(r"[^a-z0-9]+", "-", self.name.lower()).strip("-")
        s = s.replace("nvidia-", "").replace("geforce-", "")
        s = re.sub(r"-?\d+gb$", "", s)
        return f"{s}-{round(self.memory_total_gb)}gb" if self.memory_total_gb else s

    def to_dict(self) -> dict:
        d = asdict(self)
        d["slug"] = self.slug
        return d


def _nvml_detect(index: int) -> GPUInfo | None:
    try:
        import pynvml as nv

        nv.nvmlInit()
        count = nv.nvmlDeviceGetCount()
        if count == 0:
            return None
        h = nv.nvmlDeviceGetHandleByIndex(index)
        name = nv.nvmlDeviceGetName(h)
        name = name.decode() if isinstance(name, bytes) else name
        mem = nv.nvmlDeviceGetMemoryInfo(h)
        major, minor = nv.nvmlDeviceGetCudaComputeCapability(h)
        driver = nv.nvmlSystemGetDriverVersion()
        driver = driver.decode() if isinstance(driver, bytes) else driver
        try:
            power = nv.nvmlDeviceGetPowerManagementLimit(h) / 1000
        except Exception:
            power = None
        return GPUInfo(index, name, mem.total / 1024**3, f"{major}.{minor}", driver,
                       count=count, supports_bf16=major >= 8, power_limit_w=power)
    except Exception:
        return None


def _smi_detect(index: int) -> GPUInfo | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,compute_cap,driver_version",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15, check=True).stdout.strip().splitlines()
        if not out:
            return None
        name, mem, cc, drv = [x.strip() for x in out[index].split(",")]
        return GPUInfo(index, name, float(mem) / 1024, cc, drv, count=len(out),
                       supports_bf16=float(cc) >= 8.0)
    except Exception:
        return None


def _torch_detect(index: int) -> GPUInfo | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        p = torch.cuda.get_device_properties(index)
        return GPUInfo(index, p.name, p.total_memory / 1024**3, f"{p.major}.{p.minor}",
                       count=torch.cuda.device_count(), supports_bf16=p.major >= 8)
    except Exception:
        return None


def detect_gpu(index: int = 0) -> GPUInfo:
    info = _nvml_detect(index) or _smi_detect(index) or _torch_detect(index) or GPUInfo()
    try:
        import torch

        info.cuda_runtime = torch.version.cuda or ""
    except Exception:
        pass
    return info


def used_memory_gb(index: int = 0) -> float | None:
    """Aktualnie zajęta pamięć GPU (NVML) - przydatne dla vLLM, które prealokuje pamięć."""
    try:
        import pynvml as nv

        nv.nvmlInit()
        return nv.nvmlDeviceGetMemoryInfo(nv.nvmlDeviceGetHandleByIndex(index)).used / 1024**3
    except Exception:
        return None


def _version(pkg: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(pkg)
    except Exception:
        return None


def system_info() -> dict:
    import psutil

    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu": platform.processor() or platform.machine(),
        "cpu_cores": psutil.cpu_count(logical=False),
        "ram_gb": round(psutil.virtual_memory().total / 1024**3, 1),
        "packages": {p: _version(p) for p in ("torch", "transformers", "vllm", "accelerate")},
    }
