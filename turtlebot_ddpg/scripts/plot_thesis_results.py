#!/usr/bin/env python3

import argparse
import csv
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


METHOD_ORDER = [
    'geometric', 'execution_aware', 'mappo_no_vision', 'mappo_vision',
    'mappo_hybrid']
METHOD_LABELS = {
    'geometric': 'Geometric\nVoronoi',
    'execution_aware': 'Execution-\naware',
    'mappo_no_vision': 'MAPPO\n(vision masked)',
    'mappo_vision': 'MAPPO\n+ vision',
    'mappo_hybrid': 'MAPPO\n+ timed recovery',
}
COLORS = ['#4C78A8', '#F58518', '#8E6C8A', '#72B7B2', '#2E8B57']


def load_summary(path):
    with open(path, newline='') as handle:
        rows = {row['method']: row for row in csv.DictReader(handle)}
    return rows


def metric(rows, methods, field):
    return [float(rows[method][field]) for method in methods]


def main():
    parser = argparse.ArgumentParser(
        description='Plot the five-method Gazebo thesis comparison.')
    parser.add_argument('summary_csv')
    parser.add_argument('-o', '--output', default='thesis_comparison.png')
    args = parser.parse_args()

    rows = load_summary(args.summary_csv)
    methods = [method for method in METHOD_ORDER if method in rows]
    if not methods:
        raise ValueError('summary does not contain a supported method')
    labels = [METHOD_LABELS[method] for method in methods]
    colors = COLORS[:len(methods)]
    panels = [
        ('Final coverage (%)', 'coverage_pct_mean', 'coverage_pct_std'),
        ('Total path length (m)', 'path_length_total_m_mean',
         'path_length_total_m_std'),
        ('Stuck events', 'stuck_events_mean', 'stuck_events_std'),
        ('Goal completion rate', 'goal_completion_rate_mean',
         'goal_completion_rate_std'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2))
    for ax, (title, mean_field, std_field) in zip(axes.flat, panels):
        means = metric(rows, methods, mean_field)
        stds = metric(rows, methods, std_field)
        bars = ax.bar(range(len(labels)), means, yerr=stds, capsize=4,
                      color=colors, edgecolor='#333333', linewidth=0.6)
        if 'mappo_hybrid' in methods:
            recovery_index = methods.index('mappo_hybrid')
            bars[recovery_index].set_hatch('//')
            ax.axvline(recovery_index - 0.5, color='#777777',
                       linestyle=':', linewidth=1.0)
        ax.set_title(title, fontsize=11)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.grid(axis='y', color='#D9D9D9', linewidth=0.7)
        ax.set_axisbelow(True)
        upper = max(mean + std for mean, std in zip(means, stds))
        ax.set_ylim(0, upper * 1.22 if upper > 0 else 1)
        for bar, value, error in zip(bars, means, stds):
            label_y = value + error + max(upper * 0.015, 0.01)
            ax.text(bar.get_x() + bar.get_width() / 2, label_y,
                    '%.2f' % value, ha='center', va='bottom', fontsize=8)

    fig.suptitle('Gazebo Unknown-Map Multi-Robot Exploration (5 runs x 90 s)',
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(args.output, dpi=220, bbox_inches='tight')
    plt.close(fig)
    print(args.output)


if __name__ == '__main__':
    main()
