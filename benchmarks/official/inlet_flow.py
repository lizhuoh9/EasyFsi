import math
from dataclasses import dataclass

import taichi as ti

from simulation_core.runtime import TaichiRuntimeConfig, init_taichi


@dataclass(frozen=True)
class TimeWindowedInletFlow:
    volumetric_flow_m3s: float
    start_s: float
    end_s: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.volumetric_flow_m3s) or self.volumetric_flow_m3s < 0.0:
            raise ValueError("volumetric_flow_m3s must be non-negative and finite")
        if not math.isfinite(self.start_s) or not math.isfinite(self.end_s):
            raise ValueError("start_s and end_s must be finite")
        if self.end_s < self.start_s:
            raise ValueError("end_s must be greater than or equal to start_s")

    def volume_between_exact_m3(self, start_s: float, end_s: float) -> float:
        if not math.isfinite(start_s) or not math.isfinite(end_s):
            raise ValueError("integration bounds must be finite")
        if end_s < start_s:
            raise ValueError("end_s must be greater than or equal to start_s")
        overlap_s = max(0.0, min(end_s, self.end_s) - max(start_s, self.start_s))
        return self.volumetric_flow_m3s * overlap_s

    def volume_between_m3(
        self,
        start_s: float,
        end_s: float,
        *,
        sample_count: int = 4096,
        runtime_arch: str = "cuda",
    ) -> float:
        integrator = _WindowedInletVolumeIntegrator(
            sample_count=sample_count,
            runtime=TaichiRuntimeConfig(arch=runtime_arch),
        )
        return integrator.integrate(self, start_s=start_s, end_s=end_s)


@ti.data_oriented
class _WindowedInletVolumeIntegrator:
    def __init__(
        self,
        *,
        sample_count: int,
        runtime: TaichiRuntimeConfig,
    ):
        if sample_count < 16:
            raise ValueError("sample_count must be at least 16")
        init_taichi(runtime)
        self.sample_count = int(sample_count)
        self.volume_m3 = ti.field(dtype=ti.f32, shape=())

    def integrate(
        self,
        flow: TimeWindowedInletFlow,
        *,
        start_s: float,
        end_s: float,
    ) -> float:
        if not math.isfinite(start_s) or not math.isfinite(end_s):
            raise ValueError("integration bounds must be finite")
        if end_s < start_s:
            raise ValueError("end_s must be greater than or equal to start_s")
        self._integrate_kernel(
            float(flow.volumetric_flow_m3s),
            float(flow.start_s),
            float(flow.end_s),
            float(start_s),
            float(end_s),
        )
        return float(self.volume_m3[None])

    @ti.kernel
    def _integrate_kernel(
        self,
        volumetric_flow_m3s: ti.f32,
        window_start_s: ti.f32,
        window_end_s: ti.f32,
        start_s: ti.f32,
        end_s: ti.f32,
    ):
        self.volume_m3[None] = 0.0
        dt_s = (end_s - start_s) / ti.cast(self.sample_count, ti.f32)
        for index in range(self.sample_count):
            time_s = start_s + (ti.cast(index, ti.f32) + 0.5) * dt_s
            if window_start_s <= time_s <= window_end_s:
                ti.atomic_add(self.volume_m3[None], volumetric_flow_m3s * dt_s)
