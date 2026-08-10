from __future__ import annotations

import inspect
import weakref
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.refactored.validation.ansys_vertical_flap_fsi import (
    native_fine_rendering,
)
from src.refactored.validation.turek_hron_fsi import rendering as rendering_module
from src.refactored.validation.turek_hron_fsi.rendering import (
    FlowSnapshotContractError,
    _overlay_deformed_beam,
    _velocity_colormap,
    discover_flow_snapshot_paths,
    load_flow_snapshot,
    render_turek_hron_flow_gif,
)


def _snapshot_fields(*, step: int, peak_speed_mps: float) -> dict[str, np.ndarray]:
    y_centers = np.asarray([0.04, 0.20, 0.37], dtype=np.float64)
    z_centers = np.asarray([0.10, 0.65, 1.40, 2.35], dtype=np.float64)
    speed = np.linspace(
        0.0,
        peak_speed_mps,
        y_centers.size * z_centers.size,
        dtype=np.float64,
    ).reshape(y_centers.size, z_centers.size)
    obstacle = np.zeros_like(speed, dtype=np.int32)
    obstacle[1, 1] = 1

    # Two wall-normal particles at each of three beam cross-sections.  Solver
    # z decreases downstream because physical x = 2.5 - solver z.
    rest = np.asarray(
        [
            [0.025, 0.190, 2.250],
            [0.025, 0.210, 2.250],
            [0.025, 0.190, 2.075],
            [0.025, 0.210, 2.075],
            [0.025, 0.190, 1.900],
            [0.025, 0.210, 1.900],
        ],
        dtype=np.float64,
    )
    current = rest.copy()
    current[2:4, 1] += 0.005 * step
    current[4:, 1] += 0.010 * step

    return {
        "y_centers_m": y_centers,
        "z_centers_m": z_centers,
        "velocity_magnitude_yz_mps": speed,
        "velocity_y_yz_mps": np.zeros_like(speed),
        "velocity_z_yz_mps": -speed,
        "pressure_yz_pa": np.full_like(speed, float(step)),
        "obstacle_mask_yz": obstacle,
        "span_index": np.asarray(2, dtype=np.int64),
        "grid_nodes": np.asarray([4, y_centers.size, z_centers.size], dtype=np.int64),
        "cell_spacing_m": np.asarray([0.16, 0.55], dtype=np.float64),
        "beam_marker_current_xyz_m": current,
        "beam_marker_rest_xyz_m": rest,
        "beam_marker_displacement_xyz_m": current - rest,
        "time_s": np.asarray(step * 0.01, dtype=np.float64),
    }


def _write_snapshot(
    directory: Path,
    *,
    step: int,
    peak_speed_mps: float,
    drop_field: str | None = None,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    fields = _snapshot_fields(step=step, peak_speed_mps=peak_speed_mps)
    if drop_field is not None:
        fields = {key: value for key, value in fields.items() if key != drop_field}
    path = directory / f"step_{step:06d}.npz"
    np.savez(path, **fields)
    return path


def test_load_snapshot_flips_solver_z_to_ascending_physical_x(tmp_path: Path) -> None:
    path = _write_snapshot(tmp_path, step=2, peak_speed_mps=2.0)
    source = _snapshot_fields(step=2, peak_speed_mps=2.0)

    frame = load_flow_snapshot(path, channel_length_m=2.5)

    np.testing.assert_allclose(
        frame.physical_x_centers_m,
        2.5 - source["z_centers_m"][::-1],
    )
    np.testing.assert_allclose(
        frame.velocity_magnitude_mps,
        source["velocity_magnitude_yz_mps"][:, ::-1],
    )
    np.testing.assert_array_equal(
        frame.obstacle_mask,
        source["obstacle_mask_yz"][:, ::-1].astype(bool),
    )
    np.testing.assert_allclose(
        frame.beam_current_xy_m[:, 0],
        2.5 - source["beam_marker_current_xyz_m"][:, 2],
    )


def test_default_beam_overlay_is_a_black_deformed_face_without_points() -> None:
    class RecordingAxis:
        def __init__(self) -> None:
            self.fill_calls: list[dict[str, object]] = []
            self.plot_calls: list[dict[str, object]] = []
            self.scatter_calls: list[dict[str, object]] = []

        def fill(self, *args, **kwargs) -> None:
            self.fill_calls.append(dict(kwargs))

        def plot(self, *args, **kwargs) -> None:
            self.plot_calls.append(dict(kwargs))

        def scatter(self, *args, **kwargs) -> None:
            self.scatter_calls.append(dict(kwargs))

    fields = _snapshot_fields(step=2, peak_speed_mps=2.0)
    axis = RecordingAxis()

    _overlay_deformed_beam(
        axis,
        current_xyz_m=fields["beam_marker_current_xyz_m"],
        rest_xyz_m=fields["beam_marker_rest_xyz_m"],
        channel_length_m=2.5,
    )

    assert len(axis.fill_calls) == 1
    assert axis.fill_calls[0]["facecolor"] == "black"
    assert axis.fill_calls[0]["edgecolor"] == "black"
    assert axis.plot_calls == []
    assert axis.scatter_calls == []


def test_obstacle_mask_uses_opaque_black_in_velocity_colormap() -> None:
    np.testing.assert_allclose(
        _velocity_colormap().get_bad(),
        np.asarray([0.0, 0.0, 0.0, 1.0]),
    )


def test_snapshot_discovery_rejects_a_missing_periodic_step(tmp_path: Path) -> None:
    _write_snapshot(tmp_path, step=2, peak_speed_mps=1.0)
    _write_snapshot(tmp_path, step=6, peak_speed_mps=2.0)

    with pytest.raises(FlowSnapshotContractError, match="non-contiguous.*expected step 4"):
        discover_flow_snapshot_paths(tmp_path)


def test_load_snapshot_rejects_a_missing_export_field(tmp_path: Path) -> None:
    path = _write_snapshot(
        tmp_path,
        step=2,
        peak_speed_mps=2.0,
        drop_field="pressure_yz_pa",
    )

    with pytest.raises(FlowSnapshotContractError, match="pressure_yz_pa"):
        load_flow_snapshot(path)


def test_two_synthetic_snapshots_render_with_one_fixed_scale_and_make_gif(
    tmp_path: Path,
) -> None:
    snapshots = tmp_path / "flow_snapshots"
    _write_snapshot(snapshots, step=2, peak_speed_mps=1.0)
    _write_snapshot(snapshots, step=4, peak_speed_mps=3.0)

    result = render_turek_hron_flow_gif(
        snapshots,
        tmp_path / "rendered" / "velocity.gif",
        duration_ms=20,
        max_width_px=360,
        dpi=50,
    )

    assert result.steps == (2, 4)
    assert result.velocity_vmin_mps == 0.0
    assert result.velocity_vmax_mps == pytest.approx(3.0)
    assert len(result.frame_paths) == 2
    assert all(path.is_file() for path in result.frame_paths)
    assert result.gif_path.is_file()
    with Image.open(result.gif_path) as animation:
        assert animation.n_frames == 2


def test_renderer_releases_full_snapshot_before_loading_the_next(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshots = tmp_path / "flow_snapshots"
    for step in (2, 4, 6):
        _write_snapshot(snapshots, step=step, peak_speed_mps=float(step))

    original_load = rendering_module.load_flow_snapshot
    frame_references: list[weakref.ReferenceType[object]] = []
    live_before_load: list[int] = []

    def tracked_load(path: str | Path, **kwargs: object):
        live_before_load.append(sum(ref() is not None for ref in frame_references))
        frame = original_load(path, **kwargs)
        frame_references.append(weakref.ref(frame))
        return frame

    def touch_frame(_frame: object, output_path: Path, **_kwargs: object) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.touch()

    def touch_gif(
        _frame_paths: object,
        output_path: str | Path,
        **_kwargs: object,
    ) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        return path

    monkeypatch.setattr(rendering_module, "load_flow_snapshot", tracked_load)
    monkeypatch.setattr(rendering_module, "_render_velocity_frame", touch_frame)
    monkeypatch.setattr(rendering_module, "build_gif", touch_gif)

    result = rendering_module.render_turek_hron_flow_gif(
        snapshots,
        tmp_path / "rendered" / "velocity.gif",
    )

    assert result.steps == (2, 4, 6)
    assert len(live_before_load) == 6
    assert live_before_load == [0] * 6


def test_gif_encoder_writes_frames_incrementally() -> None:
    source = inspect.getsource(native_fine_rendering.build_gif)

    assert "GifImagePlugin.getdata" in source
    assert "append_images" not in source
