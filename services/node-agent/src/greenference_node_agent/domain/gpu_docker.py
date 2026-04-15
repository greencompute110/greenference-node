"""Detect the correct Docker GPU passthrough method for this host.

Docker + NVIDIA has three eras of GPU passthrough:
  1. nvidia-docker2: --runtime=nvidia + NVIDIA_VISIBLE_DEVICES env var
  2. NVIDIA Container Toolkit (legacy): --gpus flag using nvidia-container-cli
  3. NVIDIA Container Toolkit (CDI): --gpus flag using CDI device specs

Newer Docker versions (25+) default to CDI for --gpus, which fails if
`nvidia-ctk cdi generate` was never run. This module probes once at first
call and caches the working method.

When running inside Docker (typical for node-agent), the probe talks to the
host daemon via the mounted Docker socket, so the test containers run on the
host — exactly the same path real workloads take.
"""

from __future__ import annotations

import logging
import subprocess

logger = logging.getLogger(__name__)

# Cached result: "gpus", "runtime", or "env_only"
_gpu_mode: str | None = None

_TOOLKIT_IMAGE = "nvidia/cuda:12.4.1-base-ubuntu22.04"


def _run(cmd: list[str], timeout: float = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)  # noqa: S603


def _try_gpus() -> bool:
    """Test --gpus all."""
    try:
        r = _run(["docker", "run", "--rm", "--gpus", "all", _TOOLKIT_IMAGE, "nvidia-smi", "-L"])
        if r.returncode == 0:
            return True
        logger.debug("--gpus probe failed: %s", r.stderr.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        logger.debug("--gpus probe exception: %s", e)
    return False


def _try_runtime() -> bool:
    """Test --runtime=nvidia."""
    try:
        r = _run([
            "docker", "run", "--rm", "--runtime=nvidia",
            "-e", "NVIDIA_VISIBLE_DEVICES=all",
            _TOOLKIT_IMAGE, "nvidia-smi", "-L",
        ])
        if r.returncode == 0:
            return True
        logger.debug("--runtime=nvidia probe failed: %s", r.stderr.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        logger.debug("--runtime=nvidia probe exception: %s", e)
    return False


def _generate_cdi_specs() -> bool:
    """Generate CDI specs on the host by running nvidia-ctk in a privileged container.

    The nvidia/cuda image doesn't ship nvidia-ctk, but the NVIDIA Container
    Toolkit image does.  We run it with host PID namespace and the host /etc/cdi
    bind-mounted so the generated spec lands on the host.
    """
    logger.info("Attempting to generate CDI specs on host via container...")
    try:
        r = _run([
            "docker", "run", "--rm", "--privileged",
            "--pid=host",
            "-v", "/etc/cdi:/etc/cdi",
            "-v", "/var/run/docker.sock:/var/run/docker.sock",
            "nvcr.io/nvidia/k8s/container-toolkit:v1.17.5-ubuntu20.04",
            "nvidia-ctk", "cdi", "generate", "--output=/etc/cdi/nvidia.yaml",
        ], timeout=180)
        if r.returncode == 0:
            logger.info("CDI specs generated successfully")
            return True
        logger.warning("CDI generate failed (rc=%d): %s", r.returncode, r.stderr.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        logger.warning("CDI generate exception: %s", e)
    return False


def _probe_gpu_mode() -> str:
    """Try each GPU method and return the first that works."""

    # Method 1: --gpus (works if CDI is configured or on older Docker < 25)
    if _try_gpus():
        logger.info("GPU passthrough: --gpus")
        return "gpus"

    # Method 2: --runtime=nvidia (nvidia-docker2 / daemon.json configured)
    if _try_runtime():
        logger.info("GPU passthrough: --runtime=nvidia")
        return "runtime"

    # Method 3: generate CDI specs on host, then retry --gpus
    if _generate_cdi_specs() and _try_gpus():
        logger.info("GPU passthrough: --gpus (after CDI generate)")
        return "gpus"

    # Last resort
    logger.warning(
        "No GPU passthrough method verified. "
        "Run 'sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml' "
        "on the host, then restart."
    )
    return "env_only"


def get_gpu_mode() -> str:
    """Return the cached GPU mode, probing on first call."""
    global _gpu_mode
    if _gpu_mode is None:
        _gpu_mode = _probe_gpu_mode()
    return _gpu_mode


def gpu_docker_flags(device_ids: list[int] | None) -> list[str]:
    """Return Docker CLI flags to pass GPUs to a container.

    Args:
        device_ids: specific GPU device IDs, or None for all GPUs.

    Returns:
        List of CLI args to insert into ``docker run`` command.
    """
    mode = get_gpu_mode()
    device_str = ",".join(str(d) for d in device_ids) if device_ids else "all"

    if mode == "gpus":
        if device_ids:
            return ["--gpus", f"device={device_str}"]
        return ["--gpus", "all"]
    elif mode == "runtime":
        return ["--runtime=nvidia", "-e", f"NVIDIA_VISIBLE_DEVICES={device_str}"]
    else:
        return ["-e", f"NVIDIA_VISIBLE_DEVICES={device_str}"]
