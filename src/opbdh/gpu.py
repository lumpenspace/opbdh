from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GpuOffer:
    id: str
    memory_gb: int
    community_dollars_per_hour: float
    secure_dollars_per_hour: float

    def hourly(self, cloud_type: str) -> float:
        return self.community_dollars_per_hour if cloud_type.upper() == "COMMUNITY" else self.secure_dollars_per_hour


# Estimates are intentionally conservative and only used to choose a candidate
# list before RunPod performs the real availability check.
GPU_CATALOG: tuple[GpuOffer, ...] = (
    GpuOffer("NVIDIA GeForce RTX 3090", 24, 0.22, 0.70),
    GpuOffer("NVIDIA GeForce RTX 4090", 24, 0.34, 1.10),
    GpuOffer("NVIDIA L4", 24, 0.40, 1.10),
    GpuOffer("NVIDIA A40", 48, 0.74, 1.22),
    GpuOffer("NVIDIA L40", 48, 0.79, 1.90),
    GpuOffer("NVIDIA L40S", 48, 0.79, 1.90),
    GpuOffer("NVIDIA RTX 6000 Ada Generation", 48, 0.74, 1.90),
    GpuOffer("NVIDIA A100 80GB PCIe", 80, 1.19, 1.39),
    GpuOffer("NVIDIA A100-SXM4-80GB", 80, 1.39, 1.49),
    GpuOffer("NVIDIA H100 PCIe", 80, 1.99, 2.39),
    GpuOffer("NVIDIA H100 80GB HBM3", 80, 2.69, 4.00),
    GpuOffer("NVIDIA H100 NVL", 94, 2.69, 4.00),
    GpuOffer("NVIDIA RTX PRO 6000 Blackwell Server Edition", 96, 1.58, 5.58),
    GpuOffer("NVIDIA H200", 141, 3.59, 5.58),
    GpuOffer("NVIDIA B200", 180, 5.98, 8.64),
    GpuOffer("AMD Instinct MI300X OAM", 192, 3.99, 6.50),
)


def candidate_gpus(min_vram_gb: int, max_dollars_per_hour: float | None, cloud_type: str) -> list[GpuOffer]:
    cloud = cloud_type.upper()
    candidates = [gpu for gpu in GPU_CATALOG if gpu.memory_gb >= min_vram_gb]
    if max_dollars_per_hour is not None and max_dollars_per_hour > 0:
        candidates = [gpu for gpu in candidates if gpu.hourly(cloud) <= max_dollars_per_hour]
    return sorted(candidates, key=lambda gpu: (gpu.memory_gb, gpu.hourly(cloud), gpu.id))


def gpu_type_ids(min_vram_gb: int, max_dollars_per_hour: float | None, cloud_type: str) -> list[str]:
    return [gpu.id for gpu in candidate_gpus(min_vram_gb, max_dollars_per_hour, cloud_type)]


def estimated_hourly(gpu_id: str, cloud_type: str) -> float | None:
    for gpu in GPU_CATALOG:
        if gpu.id == gpu_id:
            return gpu.hourly(cloud_type)
    return None
