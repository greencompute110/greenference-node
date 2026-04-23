"""Targon-style TEE attestation and security tier detection."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from greencompute_protocol import SecurityTier


class AttestationEngine:
    """Detects available TEE hardware and generates attestation evidence."""

    def detect_security_tier(self) -> SecurityTier:
        if self._has_gpu_cc():
            return SecurityTier.CPU_GPU_ATTESTED
        if self._has_cpu_tee():
            return SecurityTier.CPU_TEE
        return SecurityTier.STANDARD

    def generate_evidence(self) -> dict[str, Any]:
        tier = self.detect_security_tier()
        evidence: dict[str, Any] = {
            "tier": tier.value,
            "platform": self._platform_info(),
        }
        if tier in {SecurityTier.CPU_TEE, SecurityTier.CPU_GPU_ATTESTED}:
            evidence["tee_type"] = self._tee_type()
            evidence["measurement"] = self._read_measurement()
        if tier == SecurityTier.CPU_GPU_ATTESTED:
            evidence["gpu_cc"] = self._gpu_cc_info()
        return evidence

    def attest_before_lease(self) -> bool:
        """Gate check: returns True if this node can accept leases (always True for STANDARD)."""
        tier = self.detect_security_tier()
        return tier in {SecurityTier.STANDARD, SecurityTier.CPU_TEE, SecurityTier.CPU_GPU_ATTESTED}

    def _has_cpu_tee(self) -> bool:
        return Path("/dev/tdx-guest").exists() or Path("/dev/sev-guest").exists() or Path("/dev/sev").exists()

    def _has_gpu_cc(self) -> bool:
        try:
            result = subprocess.run(  # noqa: S603
                ["nvidia-smi", "conf-compute", "-gccm"],  # noqa: S607
                capture_output=True,
                text=True,
                timeout=5.0,
            )
            return result.returncode == 0 and "CC" in result.stdout
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            return False

    def _tee_type(self) -> str:
        if Path("/dev/tdx-guest").exists():
            return "intel_tdx"
        if Path("/dev/sev-guest").exists() or Path("/dev/sev").exists():
            return "amd_sev_snp"
        return "unknown"

    def _read_measurement(self) -> str | None:
        for path in ("/dev/tdx-guest", "/dev/sev-guest"):
            if Path(path).exists():
                return f"attestation-evidence-from-{path}"
        return None

    def _gpu_cc_info(self) -> dict[str, Any]:
        try:
            result = subprocess.run(  # noqa: S603
                ["nvidia-smi", "--query-gpu=name,uuid", "--format=csv,noheader"],  # noqa: S607
                capture_output=True,
                text=True,
                timeout=5.0,
            )
            if result.returncode == 0:
                return {"gpus": [line.strip() for line in result.stdout.splitlines() if line.strip()]}
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            pass
        return {}

    def _platform_info(self) -> dict[str, str]:
        info: dict[str, str] = {}
        try:
            result = subprocess.run(  # noqa: S603
                ["uname", "-r"],  # noqa: S607
                capture_output=True,
                text=True,
                timeout=2.0,
            )
            if result.returncode == 0:
                info["kernel"] = result.stdout.strip()
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            pass
        return info
