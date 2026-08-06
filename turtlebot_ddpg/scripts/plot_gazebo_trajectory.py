#!/usr/bin/env python3
"""Plot robot trajectories recorded by a Gazebo evaluation metrics file."""

import argparse
import csv
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


WORLD_PROFILES = {
    'research': {
        'arena': (-15.0, -7.5, 15.0, 7.5),
        'obstacles': [
            (-10.5, 1.5, -7.5, 4.5), (-7.25, -5.25, -4.75, -2.75),
            (-2.0, 4.0, 2.0, 6.0), (2.0, -5.0, 4.0, -1.0),
            (7.5, 0.5, 10.5, 3.5), (8.75, -6.25, 11.25, -3.75)],
    },
    'hu_aligned': {
        'arena': (-10.0, -5.0, 10.0, 5.0),
        'obstacles': [
            (-6.1, -3.2, -3.9, 3.4),
            (2.4, -2.6, 5.6, 0.6),
            (5.6, 0.6, 8.4, 3.4)],
    },
    'hu_layout_b': {
        'arena': (-10.0, -5.0, 10.0, 5.0),
        'obstacles': [
            (-7.0, -2.0, -3.0, 0.0),
            (-1.0, 0.0, 1.0, 3.0),
            (4.0, -2.5, 7.0, -0.5)],
    },
    'hu_layout_c': {
        'arena': (-10.0, -5.0, 10.0, 5.0),
        'obstacles': [
            (-7.5, -3.5, -4.5, -0.5),
            (-1.5, -1.0, 1.5, 1.0),
            (4.5, 1.0, 7.5, 3.0)],
    },
}


def read_trajectories(path):
    trajectories = dict((name, []) for name in ('robot1', 'robot2', 'robot3'))
    with open(path, newline='') as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError('metrics file has no data rows')
    for row in rows:
        for name in trajectories:
            x_text, y_text = row.get(name + '_x', ''), row.get(name + '_y', '')
            if x_text and y_text:
                trajectories[name].append((float(x_text), float(y_text)))
    if not all(trajectories.values()):
        raise ValueError('metrics file does not contain all robot trajectories')
    return rows[-1], trajectories


def plot(metrics_path, output_path):
    final_row, trajectories = read_trajectories(metrics_path)
    profile_name = final_row.get('world_profile', 'hu_aligned')
    profile = WORLD_PROFILES[profile_name]
    x0, y0, x1, y1 = profile['arena']
    colors = {'robot1': '#d62728', 'robot2': '#1f77b4', 'robot3': '#2ca02c'}

    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    ax.add_patch(Rectangle(
        (x0, y0), x1 - x0, y1 - y0,
        fill=False, linewidth=2.0, edgecolor='black'))
    for ox0, oy0, ox1, oy1 in profile['obstacles']:
        ax.add_patch(Rectangle(
            (ox0, oy0), ox1 - ox0, oy1 - oy0,
            facecolor='#9a9a9a', edgecolor='#555555', linewidth=1.0))
    for name, points in trajectories.items():
        xs, ys = zip(*points)
        ax.plot(xs, ys, color=colors[name], linewidth=1.4, label=name)
        ax.scatter(xs[0], ys[0], color=colors[name], marker='o', s=35)
        ax.scatter(xs[-1], ys[-1], color=colors[name], marker='x', s=55)

    coverage = float(final_row.get('coverage_pct', 0.0) or 0.0)
    elapsed = final_row.get('elapsed_sim_s') or final_row.get('elapsed_s') or '0'
    pose_source = final_row.get('pose_source', 'unknown')
    ax.set_title(
        'Gazebo trajectory: %s, %.2f%% coverage, %.1f s, %s pose' %
        (profile_name, coverage, float(elapsed), pose_source))
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(x0 - 0.3, x1 + 0.3)
    ax.set_ylim(y0 - 0.3, y1 + 0.3)
    ax.grid(True, alpha=0.25)
    ax.legend(loc='upper center', ncol=3)
    fig.tight_layout()
    directory = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(directory, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('metrics_csv')
    parser.add_argument('output_png')
    args = parser.parse_args()
    plot(args.metrics_csv, args.output_png)


if __name__ == '__main__':
    main()
