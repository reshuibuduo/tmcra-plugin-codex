from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class GpuCapacitySnapshot:
    device: str
    total_bytes: int | None
    free_bytes: int | None
    allocated_bytes: int | None
    reserved_bytes: int | None
    reusable_reserved_bytes: int | None
    effective_free_bytes: int | None
    headroom_bytes: int
    replica_estimate_bytes: int
    can_add_replica: bool
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class CudaReplicaCapacityGuard:
    """Conservative admission guard for adding a resident model replica."""

    def __init__(
        self,
        *,
        device: str,
        headroom_bytes: int,
        replica_estimate_bytes: int,
    ) -> None:
        if headroom_bytes <= 0 or replica_estimate_bytes <= 0:
            raise ValueError("GPU capacity limits must be positive")
        self.device = device
        self.headroom_bytes = int(headroom_bytes)
        self.replica_estimate_bytes = int(replica_estimate_bytes)

    def snapshot(self) -> GpuCapacitySnapshot:
        import torch

        device = torch.device(self.device)
        if device.type != "cuda":
            return GpuCapacitySnapshot(
                device=str(device),
                total_bytes=None,
                free_bytes=None,
                allocated_bytes=None,
                reserved_bytes=None,
                reusable_reserved_bytes=None,
                effective_free_bytes=None,
                headroom_bytes=self.headroom_bytes,
                replica_estimate_bytes=self.replica_estimate_bytes,
                can_add_replica=True,
                reason="non_cuda_device",
            )
        if not torch.cuda.is_available():
            return GpuCapacitySnapshot(
                device=str(device),
                total_bytes=None,
                free_bytes=None,
                allocated_bytes=None,
                reserved_bytes=None,
                reusable_reserved_bytes=None,
                effective_free_bytes=None,
                headroom_bytes=self.headroom_bytes,
                replica_estimate_bytes=self.replica_estimate_bytes,
                can_add_replica=False,
                reason="cuda_unavailable",
            )
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        allocated_bytes = int(torch.cuda.memory_allocated(device))
        reserved_bytes = int(torch.cuda.memory_reserved(device))
        reusable_reserved = max(0, reserved_bytes - allocated_bytes)
        effective_free = int(free_bytes) + reusable_reserved
        required = self.headroom_bytes + self.replica_estimate_bytes
        allowed = effective_free >= required
        return GpuCapacitySnapshot(
            device=str(device),
            total_bytes=int(total_bytes),
            free_bytes=int(free_bytes),
            allocated_bytes=allocated_bytes,
            reserved_bytes=reserved_bytes,
            reusable_reserved_bytes=reusable_reserved,
            effective_free_bytes=effective_free,
            headroom_bytes=self.headroom_bytes,
            replica_estimate_bytes=self.replica_estimate_bytes,
            can_add_replica=allowed,
            reason="capacity_available" if allowed else "gpu_headroom_guard",
        )

    def __call__(self, current_size: int, target_size: int) -> bool:
        del current_size, target_size
        return self.snapshot().can_add_replica
