"""GPU/hardware telemetry with GPU split accounting."""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel

from greencompute_protocol import NodeCapability, SecurityTier


def _utcnow() -> datetime:
    return datetime.now(UTC)


class TelemetrySnapshot(BaseModel):
    gpu_utilization_pct: list[float] = []
    gpu_vram_used_gb: list[float] = []
    gpu_vram_total_gb: list[float] = []
    cpu_utilization_pct: float = 0.0
    memory_used_gb: float = 0.0
    memory_total_gb: float = 0.0
    observed_at: datetime = None  # type: ignore[assignment]

    def model_post_init(self, __context: Any) -> None:
        if self.observed_at is None:
            self.observed_at = _utcnow()


class TelemetryAgent:
    """Collects GPU and system metrics. Falls back to mock if nvidia-smi unavailable."""

    def __init__(self, *, gpu_count: int = 1, vram_gb_per_gpu: int = 80) -> None:
        self.gpu_count = gpu_count
        self.vram_gb_per_gpu = vram_gb_per_gpu

    def collect(self) -> TelemetrySnapshot:
        gpu_util = self._nvidia_utilization()
        vram_used = self._nvidia_vram_used()
        mem_total, mem_used = self._system_memory()
        return TelemetrySnapshot(
            gpu_utilization_pct=gpu_util if gpu_util else [0.0] * self.gpu_count,
            gpu_vram_used_gb=vram_used if vram_used else [0.0] * self.gpu_count,
            gpu_vram_total_gb=[float(self.vram_gb_per_gpu)] * self.gpu_count,
            cpu_utilization_pct=self._cpu_utilization(),
            memory_used_gb=mem_used,
            memory_total_gb=mem_total,
        )

    def available_split_units(
        self,
        gpu_count: int,
        gpu_split_units: int,
        reserved_units: int,
    ) -> float:
        """Return fractional available_gpus based on split unit accounting."""
        total_units = gpu_count * gpu_split_units
        remaining = max(0, total_units - reserved_units)
        return remaining / gpu_split_units

    def build_node_capability(
        self,
        hotkey: str,
        node_id: str,
        *,
        gpu_model: str,
        gpu_count: int,
        vram_gb_per_gpu: int,
        cpu_cores: int,
        memory_gb: int,
        performance_score: float,
        security_tier: SecurityTier,
        available_gpus: float,
        labels: dict[str, str] | None = None,
    ) -> NodeCapability:
        return NodeCapability(
            hotkey=hotkey,
            node_id=node_id,
            gpu_model=gpu_model,
            gpu_count=gpu_count,
            available_gpus=int(available_gpus),
            vram_gb_per_gpu=vram_gb_per_gpu,
            cpu_cores=cpu_cores,
            memory_gb=memory_gb,
            performance_score=performance_score,
            security_tier=security_tier,
            labels=labels or {},
        )

    def _nvidia_utilization(self) -> list[float]:
        try:
            result = subprocess.run(  # noqa: S603
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],  # noqa: S607
                capture_output=True,
                text=True,
                timeout=5.0,
            )
            if result.returncode == 0:
                return [float(line.strip()) for line in result.stdout.splitlines() if line.strip()]
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError, ValueError):
            pass
        return []

    def _nvidia_vram_used(self) -> list[float]:
        try:
            result = subprocess.run(  # noqa: S603
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],  # noqa: S607
                capture_output=True,
                text=True,
                timeout=5.0,
            )
            if result.returncode == 0:
                return [round(float(line.strip()) / 1024, 2) for line in result.stdout.splitlines() if line.strip()]
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError, ValueError):
            pass
        return []

    def _cpu_utilization(self) -> float:
        try:
            result = subprocess.run(  # noqa: S603
                ["sh", "-c", "cat /proc/stat | head -1"],  # noqa: S607
                capture_output=True,
                text=True,
                timeout=2.0,
            )
            if result.returncode == 0:
                parts = result.stdout.split()
                if len(parts) >= 5:
                    user, nice, system, idle = int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])
                    total = user + nice + system + idle
                    return round((total - idle) / total * 100, 1) if total > 0 else 0.0
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError, ValueError, IndexError):
            pass
        return 0.0

    def _system_memory(self) -> tuple[float, float]:
        try:
            meminfo = {}
            for line in open("/proc/meminfo").readlines():  # noqa: PTH123, SIM115
                parts = line.split(":")
                if len(parts) == 2:
                    meminfo[parts[0].strip()] = int(parts[1].strip().split()[0])
            total_kb = meminfo.get("MemTotal", 0)
            avail_kb = meminfo.get("MemAvailable", 0)
            total_gb = round(total_kb / 1024 / 1024, 2)
            used_gb = round((total_kb - avail_kb) / 1024 / 1024, 2)
            return total_gb, used_gb
        except (FileNotFoundError, OSError, ValueError, IndexError):
            return 0.0, 0.0
