"""Pure 2-D occupancy updates from a planar laser scan."""

from __future__ import division

import math

import numpy as np


UNKNOWN = 0
FREE = 1
OCCUPIED = 2


def world_to_grid(x, y, arena, resolution, shape):
    """Convert world coordinates to an in-bounds (ix, iy), or None."""
    ix = int(math.floor((x - arena[0]) / resolution))
    iy = int(math.floor((y - arena[1]) / resolution))
    if 0 <= ix < shape[1] and 0 <= iy < shape[0]:
        return ix, iy
    return None


def bresenham_cells(start, end):
    """Return inclusive integer cells along a line."""
    x0, y0 = start
    x1, y1 = end
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    error = dx + dy
    cells = []
    while True:
        cells.append((x0, y0))
        if x0 == x1 and y0 == y1:
            return cells
        doubled = 2 * error
        if doubled >= dy:
            error += dy
            x0 += sx
        if doubled <= dx:
            error += dx
            y0 += sy


def _mark_occupied(grid, cell, inflation_cells):
    cx, cy = cell
    radius = max(0, int(inflation_cells))
    for iy in range(max(0, cy - radius), min(grid.shape[0], cy + radius + 1)):
        for ix in range(max(0, cx - radius), min(grid.shape[1], cx + radius + 1)):
            if (ix - cx) ** 2 + (iy - cy) ** 2 <= radius ** 2:
                grid[iy, ix] = OCCUPIED


def clear_robot_footprints(grid, robot_positions, arena, resolution,
                           radius_cells=1):
    """Keep moving robots out of the persistent static occupancy map."""
    radius = max(0, int(radius_cells))
    for position in robot_positions:
        cell = world_to_grid(position[0], position[1], arena, resolution,
                             grid.shape)
        if cell is None:
            continue
        cx, cy = cell
        for iy in range(max(0, cy - radius), min(grid.shape[0], cy + radius + 1)):
            for ix in range(max(0, cx - radius), min(grid.shape[1], cx + radius + 1)):
                if (ix - cx) ** 2 + (iy - cy) ** 2 <= radius ** 2:
                    grid[iy, ix] = FREE


def update_grid_from_scan(grid, robot_xy, robot_yaw, ranges, angle_min,
                          angle_increment, range_min, range_max, arena,
                          resolution, ray_stride=1, inflation_cells=1,
                          observed_free=None):
    """Update ``grid`` in place and return the number of processed rays.

    Ground-truth pose may be supplied by Gazebo, but obstacle knowledge comes
    only from the scan. Finite returns shorter than ``range_max`` mark an
    occupied endpoint. Infinite or maximum-range returns only clear free
    space. Existing occupied cells are never cleared by later rays.
    If supplied, `observed_free` records the unique cells cleared by accepted
    rays. Pose-cell and footprint overrides do not write to this mask.
    """
    if grid.ndim != 2:
        raise ValueError('grid must be two-dimensional')
    if observed_free is not None and (
            observed_free.shape != grid.shape or observed_free.dtype != np.bool_):
        raise ValueError('observed_free must be a boolean mask with the grid shape')
    if resolution <= 0.0 or range_max <= 0.0:
        raise ValueError('resolution and range_max must be positive')
    stride = max(1, int(ray_stride))
    start = world_to_grid(robot_xy[0], robot_xy[1], arena, resolution, grid.shape)
    if start is None:
        return 0

    processed = 0
    for index in range(0, len(ranges), stride):
        raw_range = float(ranges[index])
        if np.isnan(raw_range) or raw_range < range_min:
            continue
        hit = np.isfinite(raw_range) and raw_range < range_max - 1e-3
        measured_range = min(raw_range, range_max) if np.isfinite(raw_range) else range_max
        angle = robot_yaw + angle_min + index * angle_increment
        end_x = robot_xy[0] + measured_range * math.cos(angle)
        end_y = robot_xy[1] + measured_range * math.sin(angle)
        end = world_to_grid(end_x, end_y, arena, resolution, grid.shape)
        if end is None:
            # Clip the ray just inside the configured map rectangle.
            scale = 0.98
            for _ in range(12):
                end_x = robot_xy[0] + measured_range * scale * math.cos(angle)
                end_y = robot_xy[1] + measured_range * scale * math.sin(angle)
                end = world_to_grid(end_x, end_y, arena, resolution, grid.shape)
                if end is not None:
                    break
                scale *= 0.8
        if end is None:
            continue

        cells = bresenham_cells(start, end)
        free_cells = cells[:-1] if hit else cells
        for ix, iy in free_cells:
            if grid[iy, ix] != OCCUPIED:
                grid[iy, ix] = FREE
                if observed_free is not None:
                    observed_free[iy, ix] = True
        if hit:
            _mark_occupied(grid, cells[-1], inflation_cells)
        processed += 1
    # Inflation from a very close return can cover the robot's own cell.
    # The pose cell is physically occupied by the robot and is therefore a
    # valid planning start, so it must remain free.
    grid[start[1], start[0]] = FREE
    return processed
