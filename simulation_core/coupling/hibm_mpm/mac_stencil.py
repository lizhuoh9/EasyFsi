"""Shared staggered-MAC interpolation primitives for HIBM constraints.

The interpolation and its transpose must use byte-for-byte identical support
geometry and validity renormalisation.  Keeping those calculations here avoids
the subtle ``J``/``J.T`` drift that a separate collocated marker scatter would
introduce.
"""

import taichi as ti


@ti.func
def axis_center_grid_coordinate(
    value: ti.f32,
    faces: ti.template(),
    centers: ti.template(),
    count: ti.i32,
):
    coordinate = 0.0
    if value <= centers[0]:
        half_width = ti.max(centers[0] - faces[0], 1.0e-18)
        coordinate = -0.5 * (centers[0] - value) / half_width
    elif value >= centers[count - 1]:
        half_width = ti.max(faces[count] - centers[count - 1], 1.0e-18)
        coordinate = ti.cast(count - 1, ti.f32) + 0.5 * (
            value - centers[count - 1]
        ) / half_width
    else:
        lower = 0
        upper = count - 1
        while upper - lower > 1:
            middle = (lower + upper) // 2
            if value >= centers[middle]:
                lower = middle
            else:
                upper = middle
        upper = ti.min(lower + 1, count - 1)
        distance = ti.max(centers[upper] - centers[lower], 1.0e-18)
        coordinate = ti.cast(lower, ti.f32) + (
            value - centers[lower]
        ) / distance
    return coordinate


@ti.func
def axis_backward_face_grid_coordinate(
    value: ti.f32,
    faces: ti.template(),
    count: ti.i32,
):
    """Map a coordinate onto the ``count`` stored backward-MAC faces."""

    coordinate = 0.0
    if value <= faces[0]:
        spacing = ti.max(faces[1] - faces[0], 1.0e-18)
        coordinate = (value - faces[0]) / spacing
    elif value >= faces[count - 1]:
        spacing = ti.max(faces[count - 1] - faces[count - 2], 1.0e-18)
        coordinate = ti.cast(count - 1, ti.f32) + (
            value - faces[count - 1]
        ) / spacing
    else:
        lower = 0
        upper = count - 1
        while upper - lower > 1:
            middle = (lower + upper) // 2
            if value >= faces[middle]:
                lower = middle
            else:
                upper = middle
        upper = ti.min(lower + 1, count - 1)
        spacing = ti.max(faces[upper] - faces[lower], 1.0e-18)
        coordinate = ti.cast(lower, ti.f32) + (
            value - faces[lower]
        ) / spacing
    return coordinate


@ti.func
def mac_component_grid_coordinate(
    position,
    axis: ti.i32,
    cell_face_x_m: ti.template(),
    cell_face_y_m: ti.template(),
    cell_face_z_m: ti.template(),
    cell_center_x_m: ti.template(),
    cell_center_y_m: ti.template(),
    cell_center_z_m: ti.template(),
    nx: ti.i32,
    ny: ti.i32,
    nz: ti.i32,
):
    gx = axis_center_grid_coordinate(
        position.x,
        cell_face_x_m,
        cell_center_x_m,
        nx,
    )
    gy = axis_center_grid_coordinate(
        position.y,
        cell_face_y_m,
        cell_center_y_m,
        ny,
    )
    gz = axis_center_grid_coordinate(
        position.z,
        cell_face_z_m,
        cell_center_z_m,
        nz,
    )
    if axis == 0:
        gx = axis_backward_face_grid_coordinate(position.x, cell_face_x_m, nx)
    elif axis == 1:
        gy = axis_backward_face_grid_coordinate(position.y, cell_face_y_m, ny)
    else:
        gz = axis_backward_face_grid_coordinate(position.z, cell_face_z_m, nz)
    return ti.Vector([gx, gy, gz])


@ti.func
def mac_component_stencil_base_fraction(
    position,
    axis: ti.i32,
    cell_face_x_m: ti.template(),
    cell_face_y_m: ti.template(),
    cell_face_z_m: ti.template(),
    cell_center_x_m: ti.template(),
    cell_center_y_m: ti.template(),
    cell_center_z_m: ti.template(),
    nx: ti.i32,
    ny: ti.i32,
    nz: ti.i32,
):
    coordinate = mac_component_grid_coordinate(
        position,
        axis,
        cell_face_x_m,
        cell_face_y_m,
        cell_face_z_m,
        cell_center_x_m,
        cell_center_y_m,
        cell_center_z_m,
        nx,
        ny,
        nz,
    )
    base = ti.Vector(
        [
            ti.min(ti.max(ti.floor(coordinate.x, ti.i32), 0), nx - 2),
            ti.min(ti.max(ti.floor(coordinate.y, ti.i32), 0), ny - 2),
            ti.min(ti.max(ti.floor(coordinate.z, ti.i32), 0), nz - 2),
        ]
    )
    fraction = ti.Vector(
        [
            ti.min(ti.max(coordinate.x - ti.cast(base.x, ti.f32), 0.0), 1.0),
            ti.min(ti.max(coordinate.y - ti.cast(base.y, ti.f32), 0.0), 1.0),
            ti.min(ti.max(coordinate.z - ti.cast(base.z, ti.f32), 0.0), 1.0),
        ]
    )
    return base, fraction


@ti.func
def mac_stencil_weight(fraction, oi: ti.i32, oj: ti.i32, ok: ti.i32):
    wx = 1.0 - fraction.x if oi == 0 else fraction.x
    wy = 1.0 - fraction.y if oj == 0 else fraction.y
    wz = 1.0 - fraction.z if ok == 0 else fraction.z
    return wx * wy * wz


@ti.func
def mac_component_valid_weight(
    component_face_valid_mask: ti.template(),
    position,
    axis: ti.i32,
    cell_face_x_m: ti.template(),
    cell_face_y_m: ti.template(),
    cell_face_z_m: ti.template(),
    cell_center_x_m: ti.template(),
    cell_center_y_m: ti.template(),
    cell_center_z_m: ti.template(),
    nx: ti.i32,
    ny: ti.i32,
    nz: ti.i32,
):
    base, fraction = mac_component_stencil_base_fraction(
        position,
        axis,
        cell_face_x_m,
        cell_face_y_m,
        cell_face_z_m,
        cell_center_x_m,
        cell_center_y_m,
        cell_center_z_m,
        nx,
        ny,
        nz,
    )
    valid_weight = 0.0
    for oi, oj, ok in ti.static(ti.ndrange(2, 2, 2)):
        ii = base.x + oi
        jj = base.y + oj
        kk = base.z + ok
        if (component_face_valid_mask[ii, jj, kk] & (1 << axis)) != 0:
            valid_weight += mac_stencil_weight(fraction, oi, oj, ok)
    return valid_weight


@ti.func
def sample_mac_component(
    velocity_field: ti.template(),
    component_face_valid_mask: ti.template(),
    position,
    axis: ti.i32,
    cell_face_x_m: ti.template(),
    cell_face_y_m: ti.template(),
    cell_face_z_m: ti.template(),
    cell_center_x_m: ti.template(),
    cell_center_y_m: ti.template(),
    cell_center_z_m: ti.template(),
    nx: ti.i32,
    ny: ti.i32,
    nz: ti.i32,
):
    base, fraction = mac_component_stencil_base_fraction(
        position,
        axis,
        cell_face_x_m,
        cell_face_y_m,
        cell_face_z_m,
        cell_center_x_m,
        cell_center_y_m,
        cell_center_z_m,
        nx,
        ny,
        nz,
    )
    value = 0.0
    valid_weight = 0.0
    for oi, oj, ok in ti.static(ti.ndrange(2, 2, 2)):
        ii = base.x + oi
        jj = base.y + oj
        kk = base.z + ok
        weight = mac_stencil_weight(fraction, oi, oj, ok)
        if (component_face_valid_mask[ii, jj, kk] & (1 << axis)) != 0:
            value += weight * velocity_field[ii, jj, kk][axis]
            valid_weight += weight
    if valid_weight > 1.0e-12:
        value /= valid_weight
    return value, valid_weight
