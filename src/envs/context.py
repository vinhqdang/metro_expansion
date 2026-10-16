"""Build the static scoring context (src.eval.objectives.CityContext) from a motndp City."""

from __future__ import annotations

import numpy as np

from src.eval.objectives import CityContext


def build_context(city) -> CityContext:
    # od_mx already has existing-line and excluded-segment pairs zeroed, matching what the env rewards.
    # Trip-end mass is taken from the same matrix: the zeroed pairs lie inside the existing-line reach E,
    # which A_new excludes from numerator and denominator.
    od = np.asarray(city.od_mx, dtype=float)
    mu = od.sum(axis=0) + od.sum(axis=1)
    existing = np.zeros(city.grid_size, dtype=bool)
    for line in city.existing_lines_full:
        existing[np.asarray(line).reshape(-1).astype(int)] = True
    return CityContext(
        od_raw=od,
        mu=mu.reshape(city.grid_x_size, city.grid_y_size),
        existing_mask=existing.reshape(city.grid_x_size, city.grid_y_size),
        group_od_sum=np.asarray(city.group_od_sum, dtype=float),
        total_od=float(od.sum()),
        grid_shape=(city.grid_x_size, city.grid_y_size),
    )
