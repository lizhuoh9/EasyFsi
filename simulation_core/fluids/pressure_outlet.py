def pressure_outlet_cleanup_iteration_budget(
    *,
    iterations: int,
    pressure_interface_matrix_active: bool,
) -> int:
    iteration_count = max(1, int(iterations))
    if bool(pressure_interface_matrix_active):
        return iteration_count
    return min(iteration_count, 256)
