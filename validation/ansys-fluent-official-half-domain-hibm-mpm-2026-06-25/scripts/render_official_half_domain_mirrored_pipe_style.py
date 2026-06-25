from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


FIELDS = Path(os.environ["FIELDS_NPZ"]).resolve()
OUT_DIR = Path(os.environ.get("PIPE_STYLE_RENDER_OUT_DIR", str(FIELDS.parent))).resolve()
OUT_DIR.mkdir(parents=True, exist_ok=True)
PNG = OUT_DIR / f"{FIELDS.stem}_mirrored_velocity_pipe_style.png"
SUMMARY = OUT_DIR / f"{FIELDS.stem}_mirrored_velocity_pipe_style.json"

DUCT_LENGTH_M = 0.10
FULL_DUCT_HEIGHT_M = 0.04
MODELED_HALF_HEIGHT_M = 0.02
INLET_VELOCITY_MPS = 10.0
FLIP_Z_FOR_REFERENCE = os.environ.get("FLIP_Z_FOR_REFERENCE", "1") != "0"


def turbo_colormap(values: np.ndarray) -> np.ndarray:
    x = np.clip(values, 0.0, 1.0)
    r = (
        0.13572138
        + 4.61539260 * x
        - 42.66032258 * x**2
        + 132.13108234 * x**3
        - 152.94239396 * x**4
        + 59.28637943 * x**5
    )
    g = (
        0.09140261
        + 2.19418839 * x
        + 4.84296658 * x**2
        - 14.18503333 * x**3
        + 4.27729857 * x**4
        + 2.82956604 * x**5
    )
    b = (
        0.10667330
        + 12.64194608 * x
        - 60.58204836 * x**2
        + 110.36276771 * x**3
        - 89.90310912 * x**4
        + 27.34824973 * x**5
    )
    return (np.stack([r, g, b], axis=-1).clip(0.0, 1.0) * 255.0).astype(np.uint8)


def scientific_label(value: float) -> str:
    return f"{value:.2e}"


data = np.load(FIELDS)
velocity = np.asarray(data["velocity_mps"], dtype=np.float64)
obstacle = np.asarray(data["obstacle"], dtype=np.int32)
grid_nodes = tuple(int(v) for v in data["grid_nodes"])
x_index = grid_nodes[0] // 2

half_speed = np.linalg.norm(velocity, axis=3)[x_index, :, :]
half_solid = obstacle[x_index, :, :] != 0
full_speed = np.concatenate([half_speed, half_speed[::-1, :]], axis=0)
full_solid = np.concatenate([half_solid, half_solid[::-1, :]], axis=0)
if FLIP_Z_FOR_REFERENCE:
    full_speed = full_speed[:, ::-1]
    full_solid = full_solid[:, ::-1]

speed_img = np.flipud(full_speed)
solid_img = np.flipud(full_solid)

vmax = float(np.percentile(speed_img, 99.9))
if vmax <= 0.0:
    vmax = float(speed_img.max(initial=1.0))
rgb = turbo_colormap(speed_img / max(vmax, 1.0e-12))
field_raw = Image.fromarray(rgb, mode="RGB")
solid_raw = Image.fromarray((solid_img.astype(np.uint8) * 255), mode="L")

field_width = 980
field_height = int(round(field_width * FULL_DUCT_HEIGHT_M / DUCT_LENGTH_M))
field = field_raw.resize((field_width, field_height), Image.Resampling.BICUBIC)
solid_mask = solid_raw.resize((field_width, field_height), Image.Resampling.NEAREST)
field.paste(Image.new("RGB", field.size, "white"), mask=solid_mask)

bg = (226, 231, 240)
border_blue = (47, 101, 255)
centerline = (191, 185, 124)

canvas = Image.new("RGB", (1230, 680), bg)
draw = ImageDraw.Draw(canvas)
font = ImageFont.load_default()

bar_x, bar_y = 18, 28
bar_w, bar_h = 31, 460
plot_x, plot_y = 210, 150

canvas.paste(field, (plot_x, plot_y))
draw.rectangle([plot_x, plot_y, plot_x + field_width, plot_y + field_height], outline=border_blue, width=2)
draw.line(
    [(plot_x, plot_y + field_height // 2), (plot_x + field_width, plot_y + field_height // 2)],
    fill=centerline,
    width=2,
)

grad = np.linspace(1.0, 0.0, bar_h).reshape(bar_h, 1)
bar_img = Image.fromarray(turbo_colormap(np.repeat(grad, bar_w, axis=1)), mode="RGB")
canvas.paste(bar_img, (bar_x, bar_y))
draw.rectangle([bar_x, bar_y, bar_x + bar_w, bar_y + bar_h], outline=(120, 120, 120), width=1)
for idx, value in enumerate(np.linspace(vmax, 0.0, 11)):
    y = int(round(bar_y + idx * bar_h / 10))
    draw.line([(bar_x + bar_w + 2, y), (bar_x + bar_w + 9, y)], fill=(80, 80, 80), width=1)
    draw.text((bar_x + bar_w + 16, y - 7), scientific_label(float(value)), fill=(70, 70, 70), font=font)

summary = {
    "fields_npz": str(FIELDS),
    "png": str(PNG),
    "model": "ANSYS Fluent official half-domain, mirrored for display",
    "solver": "local HIBM-MPM advance_hibm_mpm_sharp_mpm_step",
    "case": "ansys-fluent-official-half-domain-single-flap",
    "official_half_domain": True,
    "full_domain_two_flap": False,
    "grid_nodes_modeled_half": list(grid_nodes),
    "modeled_grid_nodes": list(grid_nodes),
    "display_grid_after_symmetry_mirror": [grid_nodes[0], 2 * grid_nodes[1], grid_nodes[2]],
    "x_index": int(x_index),
    "duct_length_m": DUCT_LENGTH_M,
    "full_duct_height_m": FULL_DUCT_HEIGHT_M,
    "modeled_half_height_m": MODELED_HALF_HEIGHT_M,
    "inlet_velocity_mps": INLET_VELOCITY_MPS,
    "flip_z_for_reference": FLIP_Z_FOR_REFERENCE,
    "flap_count_modeled": 1,
    "flap_count_displayed": 2,
    "flap_count_displayed_after_symmetry_mirror": 2,
    "marker_count_actual": int(data["marker_position_m"].shape[0]),
    "speed_max_mps": float(speed_img.max(initial=0.0)),
    "speed_p99_mps": float(np.percentile(speed_img, 99.0)),
    "speed_p999_mps": vmax,
}
canvas.save(PNG)
SUMMARY.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2, sort_keys=True))
