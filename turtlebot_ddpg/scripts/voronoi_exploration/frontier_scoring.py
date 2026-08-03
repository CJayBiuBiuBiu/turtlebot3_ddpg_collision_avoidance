#!/usr/bin/env python3
"""Pure frontier-cost helpers shared by Gazebo baselines and future MAPPO."""

from __future__ import division

import numpy as np


SELECTION_MODES = ("geometric", "vision", "execution_aware")


DEFAULT_WEIGHTS = {
    "distance": 0.30,
    "path": 0.25,
    "obstacle": 0.10,
    "collision": 0.10,
    "overlap": 0.10,
    "communication": 0.05,
    "information": 0.07,
    "vision": 0.03,
}


def line_risk_path_cost(grid, start_cell, goal_cell, grid_resolution=1.0,
                        unknown_penalty=0.5, occupied_penalty=4.0):
    """Estimate execution cost along a straight grid ray.

    Distance remains the base cost. Unknown cells increase uncertainty, while
    occupied cells add a larger metric penalty that approximates a required
    detour. This is much cheaper than running A* for every frontier at control
    rate.
    """
    grid = np.asarray(grid)
    if grid.ndim != 2:
        raise ValueError('grid must be two-dimensional')
    if grid_resolution <= 0.0:
        raise ValueError('grid_resolution must be positive')
    if unknown_penalty < 0.0 or occupied_penalty < 0.0:
        raise ValueError('path penalties must be non-negative')

    x0, y0 = int(start_cell[0]), int(start_cell[1])
    x1, y1 = int(goal_cell[0]), int(goal_cell[1])
    if not (0 <= x0 < grid.shape[1] and 0 <= x1 < grid.shape[1] and
            0 <= y0 < grid.shape[0] and 0 <= y1 < grid.shape[0]):
        raise ValueError('start_cell and goal_cell must lie inside the grid')

    cells = []
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    error = dx - dy
    x, y = x0, y0
    while True:
        cells.append((x, y))
        if x == x1 and y == y1:
            break
        doubled = 2 * error
        if doubled > -dy:
            error -= dy
            x += sx
        if doubled < dx:
            error += dx
            y += sy

    direct_distance = float(np.hypot(x1 - x0, y1 - y0) * grid_resolution)
    if direct_distance == 0.0:
        return 0.0
    values = np.asarray([grid[y, x] for x, y in cells[1:]])
    sample_count = max(1, values.size)
    unknown_fraction = float(np.sum(values == 0)) / sample_count
    occupied_count = int(np.sum(values == 2))
    return (direct_distance * (1.0 + unknown_penalty * unknown_fraction) +
            occupied_penalty * grid_resolution * occupied_count)


def normalize_rows(values, finite_fallback=1.0):
    """Normalize each robot row to [0, 1] without masking usable actions.

    A row with equal finite values becomes all zero. Non-finite entries receive
    ``finite_fallback`` instead of remaining infinite; this prevents the actor
    or rule baseline from accidentally losing every frontier action.
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
        squeeze = True
    elif arr.ndim == 2:
        squeeze = False
    else:
        raise ValueError("values must be a 1-D or 2-D array")

    result = np.empty_like(arr)
    for row_idx, row in enumerate(arr):
        finite = np.isfinite(row)
        if not finite.any():
            result[row_idx] = finite_fallback
            continue
        low = np.min(row[finite])
        high = np.max(row[finite])
        if high - low < 1e-9:
            result[row_idx] = 0.0
        else:
            result[row_idx] = (row - low) / (high - low)
        result[row_idx, ~finite] = finite_fallback
    return result[0] if squeeze else result


def normalize_global(values, finite_fallback=1.0):
    """Normalize one matrix with a shared scale for cross-robot comparison."""
    arr = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.full_like(arr, finite_fallback)
    low = np.min(arr[finite])
    high = np.max(arr[finite])
    if high - low < 1e-9:
        result = np.zeros_like(arr)
    else:
        result = (arr - low) / (high - low)
    result[~finite] = finite_fallback
    return result


def coverage_phase_weight(coverage_pct, start_coverage=40.0,
                          full_coverage=90.0):
    """Return a bounded progress weight for phase-adaptive exploration.

    Early exploration benefits from simple geometric dispersion. Execution
    risk matters more once only difficult frontier pockets remain. The linear
    schedule keeps those roles explicit and makes the transition reproducible.
    """
    start = float(start_coverage)
    full = float(full_coverage)
    if full <= start:
        raise ValueError("full_coverage must be greater than start_coverage")
    return float(np.clip((float(coverage_pct) - start) / (full - start),
                         0.0, 1.0))


def blend_phase_costs(geometric_costs, execution_costs, coverage_pct,
                      start_coverage=40.0, full_coverage=90.0,
                      cross_robot=False):
    """Blend geometric and execution-aware costs by exploration progress.

    ``cross_robot=True`` uses a shared scale so the result remains valid for
    frontier ownership. Row normalization is used for one robot ranking its
    own candidates.
    """
    geometric = np.asarray(geometric_costs, dtype=np.float64)
    execution = np.asarray(execution_costs, dtype=np.float64)
    if geometric.shape != execution.shape:
        raise ValueError("geometric_costs and execution_costs must match")
    normalize = normalize_global if cross_robot else normalize_rows
    phase = coverage_phase_weight(
        coverage_pct, start_coverage=start_coverage,
        full_coverage=full_coverage)
    blended = ((1.0 - phase) * normalize(geometric) +
               phase * normalize(execution))
    return np.nan_to_num(blended, nan=1.0, posinf=1.0, neginf=-1.0), phase


def combine_cost_terms(mode, distance, path=None, obstacle=None,
                       collision=None, overlap=None, communication=None,
                       information=None, vision=None, weights=None):
    """Return a finite robot-by-frontier cost matrix for one policy mode."""
    if mode not in SELECTION_MODES:
        raise ValueError("unknown selection mode: %s" % mode)

    distance = np.asarray(distance, dtype=np.float64)
    if distance.ndim != 2:
        raise ValueError("distance must have shape [robots, frontiers]")
    shape = distance.shape

    def term(value, default=0.0):
        if value is None:
            return np.full(shape, default, dtype=np.float64)
        value = np.asarray(value, dtype=np.float64)
        return np.broadcast_to(value, shape).copy()

    distance_n = normalize_rows(distance)
    if mode == "geometric":
        return distance_n

    vision_n = np.clip(term(vision), 0.0, 1.0)
    if mode == "vision":
        return distance_n - 0.25 * vision_n

    cfg = dict(DEFAULT_WEIGHTS)
    if weights:
        cfg.update(weights)

    cost = (
        cfg["distance"] * distance_n
        + cfg["path"] * normalize_rows(term(path, finite_fallback_value(distance)))
        + cfg["obstacle"] * np.clip(term(obstacle), 0.0, 1.0)
        + cfg["collision"] * np.clip(term(collision), 0.0, 1.0)
        + cfg["overlap"] * np.clip(term(overlap), 0.0, 1.0)
        + cfg["communication"] * np.clip(term(communication), 0.0, 1.0)
        - cfg["information"] * np.clip(term(information), 0.0, 1.0)
        - cfg["vision"] * vision_n
    )
    return np.nan_to_num(cost, nan=1.0, posinf=1.0, neginf=-1.0)


def combine_assignment_cost_terms(distance, path=None, obstacle=None,
                                  collision=None, overlap=None,
                                  communication=None, information=None,
                                  vision=None, weights=None):
    """Build ownership costs whose rows remain comparable across robots.

    ``combine_cost_terms`` deliberately normalizes each robot row because it
    ranks that robot's own candidate frontiers. Ownership instead takes an
    argmin across robot rows, so row-wise normalization can assign a frontier
    to a physically farther robot. This helper uses one shared matrix scale.
    """
    distance = np.asarray(distance, dtype=np.float64)
    if distance.ndim != 2:
        raise ValueError("distance must have shape [robots, frontiers]")
    shape = distance.shape

    def term(value, default=0.0):
        if value is None:
            return np.full(shape, default, dtype=np.float64)
        value = np.asarray(value, dtype=np.float64)
        return np.broadcast_to(value, shape).copy()

    cfg = dict(DEFAULT_WEIGHTS)
    if weights:
        cfg.update(weights)
    vision_n = np.clip(term(vision), 0.0, 1.0)
    cost = (
        cfg["distance"] * normalize_global(distance)
        + cfg["path"] * normalize_global(
            term(path, finite_fallback_value(distance)))
        + cfg["obstacle"] * np.clip(term(obstacle), 0.0, 1.0)
        + cfg["collision"] * np.clip(term(collision), 0.0, 1.0)
        + cfg["overlap"] * np.clip(term(overlap), 0.0, 1.0)
        + cfg["communication"] * np.clip(term(communication), 0.0, 1.0)
        - cfg["information"] * np.clip(term(information), 0.0, 1.0)
        - cfg["vision"] * vision_n
    )
    return np.nan_to_num(cost, nan=1.0, posinf=1.0, neginf=-1.0)


def finite_fallback_value(distance):
    """Choose a finite path fallback larger than the observed distances."""
    finite = np.asarray(distance, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.max(finite) * 1.5 + 1.0) if finite.size else 1.0


def assign_by_cost(cost_matrix):
    """Assign every frontier to its lowest-cost robot."""
    costs = np.asarray(cost_matrix, dtype=np.float64)
    if costs.ndim != 2:
        raise ValueError("cost_matrix must have shape [robots, frontiers]")
    if costs.shape[1] == 0:
        return np.empty((0,), dtype=np.int64)
    safe = np.nan_to_num(costs, nan=1e6, posinf=1e6, neginf=-1e6)
    return np.argmin(safe, axis=0)
