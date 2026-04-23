"""JSON-persisted state store for unified node agent runtimes."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from greencompute_protocol import (
    CollateralRecord,
    ComputePlacementRecord,
    UnifiedRuntimeRecord,
    VolumeRecord,
)

logger = logging.getLogger(__name__)


class NodeAgentRepository:
    def __init__(self, state_path: str = "/tmp/greencompute-node-runtime-state.json") -> None:
        self.state_path = state_path
        self.runtimes: dict[str, UnifiedRuntimeRecord] = {}
        self.placements: dict[str, ComputePlacementRecord] = {}
        self.volumes: dict[str, VolumeRecord] = {}
        self.collateral: dict[str, CollateralRecord] = {}
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self.state_path):
            return
        try:
            with open(self.state_path) as f:
                data = json.load(f)
            for key, raw in data.get("runtimes", {}).items():
                self.runtimes[key] = UnifiedRuntimeRecord.model_validate(raw)
            for key, raw in data.get("placements", {}).items():
                self.placements[key] = ComputePlacementRecord.model_validate(raw)
            for key, raw in data.get("volumes", {}).items():
                self.volumes[key] = VolumeRecord.model_validate(raw)
            for key, raw in data.get("collateral", {}).items():
                self.collateral[key] = CollateralRecord.model_validate(raw)
            logger.info("loaded %d runtimes from %s", len(self.runtimes), self.state_path)
        except Exception:
            logger.exception("failed to load state from %s", self.state_path)

    def save(self) -> None:
        data: dict[str, Any] = {
            "runtimes": {k: v.model_dump(mode="json") for k, v in self.runtimes.items()},
            "placements": {k: v.model_dump(mode="json") for k, v in self.placements.items()},
            "volumes": {k: v.model_dump(mode="json") for k, v in self.volumes.items()},
            "collateral": {k: v.model_dump(mode="json") for k, v in self.collateral.items()},
        }
        tmp = self.state_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f)
        os.replace(tmp, self.state_path)

    def upsert_runtime(self, runtime: UnifiedRuntimeRecord) -> UnifiedRuntimeRecord:
        self.runtimes[runtime.deployment_id] = runtime
        self.save()
        return runtime

    def get_runtime(self, deployment_id: str) -> UnifiedRuntimeRecord | None:
        return self.runtimes.get(deployment_id)

    def remove_runtime(self, deployment_id: str) -> None:
        self.runtimes.pop(deployment_id, None)
        self.save()

    def runtime_summary(self) -> dict[str, Any]:
        by_status: dict[str, int] = {}
        by_kind: dict[str, int] = {}
        failed = 0
        for rt in self.runtimes.values():
            by_status[rt.status] = by_status.get(rt.status, 0) + 1
            by_kind[rt.workload_kind] = by_kind.get(rt.workload_kind, 0) + 1
            if rt.status == "failed":
                failed += 1
        return {
            "total": len(self.runtimes),
            "by_status": by_status,
            "by_kind": by_kind,
            "failed": failed,
        }
