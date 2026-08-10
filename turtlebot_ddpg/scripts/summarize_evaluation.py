#!/usr/bin/env python3
"""Summarize final rows from paired Gazebo evaluation CSV files."""

import argparse
import csv
import glob
import math
import os
import statistics


METRICS = [
    'elapsed_s', 'elapsed_wall_s', 'observed_real_time_factor',
    'mission_done', 'evaluation_horizon_reached', 'coverage_pct',
    'path_length_total_m', 'goals_reached',
    'goal_assignments', 'no_path_fallbacks',
    'no_path_start_occupied', 'no_path_goal_occupied',
    'no_path_disconnected', 'stuck_events', 'unique_stuck_regions',
    'recovery_results', 'recovery_successes', 'recovery_success_rate',
    'recovery_mean_displacement_m',
    'near_collision_events', 'near_collision_active_s',
    'near_collision_sample_ratio',
    'policy_actions_received',
    'vision_image_mean_openness', 'vision_token_mean_openness',
    'vision_token_nonzero_ratio', 'vision_observation_mean_openness',
    'vision_observation_std_openness', 'vision_direction_contrast_mean',
    'policy_select_requests', 'policy_select_accepted',
    'policy_select_stale', 'policy_select_owner_rejected',
    'policy_action_timeouts', 'goals_dropped_owner',
    'goals_dropped_blocked', 'goals_cleared_wait',
    'goals_replaced_policy', 'hybrid_fallback_activations',
    'hybrid_fallback_goal_assignments',
    'adaptive_execution_weight', 'tail_completion_active',
    'tail_completion_activations', 'tail_completion_goal_assignments',
]

PAIRED_METRICS = [
    'elapsed_s', 'time_to_80_s', 'time_to_90_s',
    'time_to_95_s', 'time_to_97_s', 'time_90_to_97_s',
    'mean_coverage_0_90_pct', 'mean_coverage_0_300_pct',
    'coverage_pct', 'path_length_total_m', 'goals_reached',
    'goal_assignments', 'goal_completion_rate', 'stuck_events',
    'unique_stuck_regions', 'recovery_success_rate',
    'recovery_mean_displacement_m',
    'near_collision_events', 'near_collision_active_s',
    'near_collision_sample_ratio', 'coverage_per_meter',
]
THRESHOLDS = (80, 90, 95, 97)
EARLY_COVERAGE_HORIZON_S = 300.0
LOWER_IS_BETTER = {
    'elapsed_s', 'time_to_80_s', 'time_to_90_s', 'time_to_95_s',
    'time_to_97_s', 'time_90_to_97_s', 'path_length_total_m',
    'stuck_events', 'unique_stuck_regions', 'near_collision_events',
    'near_collision_active_s', 'near_collision_sample_ratio',
}
HIGHER_IS_BETTER = {
    'mean_coverage_0_90_pct', 'mean_coverage_0_300_pct', 'coverage_pct',
    'goals_reached', 'goal_completion_rate', 'coverage_per_meter',
    'recovery_success_rate', 'recovery_mean_displacement_m',
}


def analysis_elapsed(record):
    """Use Gazebo simulation time when present, else legacy wall time."""
    simulation_time = record.get('elapsed_sim_s', '')
    if simulation_time not in ('', None):
        return float(simulation_time)
    return float(record.get('elapsed_s', 0.0) or 0.0)


def time_weighted_coverage(records, horizon_s=EARLY_COVERAGE_HORIZON_S):
    """Return mean coverage over a fixed horizon using trapezoidal AUC."""
    samples = sorted(
        (analysis_elapsed(record),
         float(record.get('coverage_pct', 0.0) or 0.0))
        for record in records)
    if not samples:
        return math.nan

    points = [(0.0, 0.0)]
    previous_time, previous_coverage = points[0]
    for elapsed, coverage in samples:
        if elapsed <= previous_time:
            previous_coverage = coverage
            points[-1] = (previous_time, previous_coverage)
            continue
        if elapsed >= horizon_s:
            fraction = ((horizon_s - previous_time) /
                        max(elapsed - previous_time, 1e-9))
            interpolated = previous_coverage + fraction * (
                coverage - previous_coverage)
            points.append((horizon_s, interpolated))
            break
        points.append((elapsed, coverage))
        previous_time, previous_coverage = elapsed, coverage
    if points[-1][0] < horizon_s:
        points.append((horizon_s, points[-1][1]))

    area = 0.0
    for (time_a, coverage_a), (time_b, coverage_b) in zip(points, points[1:]):
        area += 0.5 * (coverage_a + coverage_b) * (time_b - time_a)
    return area / horizon_s


def t_critical_95(sample_count):
    """Return a two-sided 95% Student-t critical value for small samples."""
    critical_by_df = {
        1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776,
        5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
        9: 2.262, 10: 2.228, 15: 2.131, 20: 2.086,
        30: 2.042,
    }
    degrees_of_freedom = max(1, int(sample_count) - 1)
    for upper_df in sorted(critical_by_df):
        if degrees_of_freedom <= upper_df:
            return critical_by_df[upper_df]
    return 1.96


def exact_two_sided_sign_test(positive_better, negative_better):
    """Return the exact two-sided sign-test p-value, excluding ties."""
    sample_count = int(positive_better) + int(negative_better)
    if sample_count <= 0:
        return math.nan
    tail_count = min(int(positive_better), int(negative_better))
    tail_probability = sum(
        math.comb(sample_count, index) for index in range(tail_count + 1))
    return min(1.0, 2.0 * tail_probability / float(2 ** sample_count))


def read_final_rows(directory):
    rows = []
    for path in sorted(glob.glob(os.path.join(directory, 'repeat_*.csv'))):
        with open(path, newline='') as stream:
            records = list(csv.DictReader(stream))
        if not records:
            continue
        record = records[-1]
        uses_sim_time = any(
            candidate.get('elapsed_sim_s', '') not in ('', None)
            for candidate in records)
        record['time_basis'] = (
            'gazebo_sim_time' if uses_sim_time else 'legacy_wall_time')
        record['elapsed_wall_s'] = (
            record.get('elapsed_wall_s', '') or record.get('elapsed_s', ''))
        record['elapsed_s'] = analysis_elapsed(record)
        for threshold in THRESHOLDS:
            crossing = next((candidate for candidate in records
                             if float(candidate.get('coverage_pct', 0.0) or 0.0)
                             >= threshold), None)
            record['time_to_%d_s' % threshold] = (
                analysis_elapsed(crossing) if crossing is not None else '')
        time_to_90 = record.get('time_to_90_s', '')
        time_to_97 = record.get('time_to_97_s', '')
        record['time_90_to_97_s'] = (
            float(time_to_97) - float(time_to_90)
            if time_to_90 not in ('', None) and time_to_97 not in ('', None)
            else '')
        record['mean_coverage_0_90_pct'] = time_weighted_coverage(records, 90.0)
        record['mean_coverage_0_300_pct'] = time_weighted_coverage(records, 300.0)
        record['_file'] = os.path.basename(path)
        filename_parts = os.path.splitext(record['_file'])[0].split('_', 2)
        if len(filename_parts) == 3:
            record['selection_mode'] = filename_parts[2]
        rows.append(record)
    return rows


def summarize(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row['selection_mode'], []).append(row)
    output = []
    for method, method_rows in sorted(grouped.items()):
        time_bases = sorted(set(
            row.get('time_basis', 'legacy_wall_time') for row in method_rows))
        summary = {
            'method': method,
            'runs': len(method_rows),
            'time_basis': ';'.join(time_bases),
        }
        for metric in METRICS:
            values = [float(row.get(metric, 0.0) or 0.0) for row in method_rows]
            summary[metric + '_mean'] = statistics.mean(values)
            summary[metric + '_std'] = statistics.stdev(values) if len(values) > 1 else 0.0
        for horizon in (90, 300):
            field = 'mean_coverage_0_%d_pct' % horizon
            early_coverage = [float(row.get(field, 0.0) or 0.0)
                              for row in method_rows]
            summary[field + '_mean'] = statistics.mean(early_coverage)
            summary[field + '_std'] = (
                statistics.stdev(early_coverage)
                if len(early_coverage) > 1 else 0.0)
        requests = summary['policy_select_requests_mean']
        accepted = summary['policy_select_accepted_mean']
        summary['policy_acceptance_rate_mean'] = (
            accepted / requests if requests > 0.0 else math.nan)
        completion_rates = []
        coverage_efficiencies = []
        replacement_rates = []
        for row in method_rows:
            assignments = float(row.get('goal_assignments', 0.0) or 0.0)
            reached = float(row.get('goals_reached', 0.0) or 0.0)
            path_length = float(row.get('path_length_total_m', 0.0) or 0.0)
            coverage = float(row.get('coverage_pct', 0.0) or 0.0)
            replacements = float(row.get('goals_replaced_policy', 0.0) or 0.0)
            accepted_actions = float(row.get('policy_select_accepted', 0.0) or 0.0)
            completion_rates.append(reached / assignments if assignments > 0.0 else 0.0)
            coverage_efficiencies.append(coverage / path_length if path_length > 0.0 else 0.0)
            replacement_rates.append(
                replacements / accepted_actions if accepted_actions > 0.0 else 0.0)
        summary['goal_completion_rate_mean'] = statistics.mean(completion_rates)
        summary['goal_completion_rate_std'] = (
            statistics.stdev(completion_rates) if len(completion_rates) > 1 else 0.0)
        summary['coverage_per_meter_mean'] = statistics.mean(coverage_efficiencies)
        summary['coverage_per_meter_std'] = (
            statistics.stdev(coverage_efficiencies)
            if len(coverage_efficiencies) > 1 else 0.0)
        summary['policy_replacement_rate_mean'] = statistics.mean(replacement_rates)
        for threshold in THRESHOLDS:
            field = 'time_to_%d_s' % threshold
            times = [float(row[field]) for row in method_rows
                     if row.get(field, '') not in ('', None)]
            summary['success_rate_%d' % threshold] = (
                float(len(times)) / len(method_rows))
            summary[field + '_mean'] = statistics.mean(times) if times else math.nan
            summary[field + '_std'] = (
                statistics.stdev(times) if len(times) > 1 else 0.0
                if times else math.nan)
        tail_times = [float(row['time_90_to_97_s']) for row in method_rows
                      if row.get('time_90_to_97_s', '') not in ('', None)]
        summary['time_90_to_97_s_mean'] = (
            statistics.mean(tail_times) if tail_times else math.nan)
        summary['time_90_to_97_s_std'] = (
            statistics.stdev(tail_times) if len(tail_times) > 1 else 0.0
            if tail_times else math.nan)
        output.append(summary)
    return output


def add_derived_metrics(row):
    derived = dict(row)
    assignments = float(row.get('goal_assignments', 0.0) or 0.0)
    reached = float(row.get('goals_reached', 0.0) or 0.0)
    path_length = float(row.get('path_length_total_m', 0.0) or 0.0)
    coverage = float(row.get('coverage_pct', 0.0) or 0.0)
    derived['goal_completion_rate'] = reached / assignments if assignments > 0.0 else 0.0
    derived['coverage_per_meter'] = coverage / path_length if path_length > 0.0 else 0.0
    return derived


def paired_differences(rows, positive_method='mappo',
                       negative_method='execution_aware'):
    paired = {}
    for row in rows:
        parts = row['_file'].replace('.csv', '').split('_')
        repeat = int(parts[1])
        paired.setdefault(repeat, {})[row['selection_mode']] = add_derived_metrics(row)

    output = []
    for metric in PAIRED_METRICS:
        differences = []
        experiment_seeds = []
        for repeat in sorted(paired):
            methods = paired[repeat]
            if positive_method not in methods or negative_method not in methods:
                continue
            positive_value = methods[positive_method].get(metric, '')
            negative_value = methods[negative_method].get(metric, '')
            if positive_value in ('', None) or negative_value in ('', None):
                continue
            positive_seed = methods[positive_method].get('experiment_seed', '')
            negative_seed = methods[negative_method].get('experiment_seed', '')
            if (positive_seed not in ('', None) and
                    negative_seed not in ('', None) and
                    positive_seed != negative_seed):
                raise ValueError(
                    'repeat %d is not seed-matched: %s uses %s, %s uses %s' %
                    (repeat, positive_method, positive_seed,
                     negative_method, negative_seed))
            if positive_seed not in ('', None):
                experiment_seeds.append(str(positive_seed))
            differences.append(
                float(positive_value) - float(negative_value))
        if not differences:
            continue
        mean = statistics.mean(differences)
        std = statistics.stdev(differences) if len(differences) > 1 else 0.0
        half_width = (
            t_critical_95(len(differences)) * std / math.sqrt(len(differences))
            if len(differences) > 1 else 0.0)
        if metric in LOWER_IS_BETTER:
            preferred_direction = 'lower'
            positive_better = sum(value < 0.0 for value in differences)
            negative_better = sum(value > 0.0 for value in differences)
        elif metric in HIGHER_IS_BETTER:
            preferred_direction = 'higher'
            positive_better = sum(value > 0.0 for value in differences)
            negative_better = sum(value < 0.0 for value in differences)
        else:
            preferred_direction = 'descriptive'
            positive_better = 0
            negative_better = 0
        ties = sum(value == 0.0 for value in differences)
        output.append({
            'metric': metric,
            'pairs': len(differences),
            'positive_method': positive_method,
            'negative_method': negative_method,
            'positive_minus_negative_mean': mean,
            'paired_std': std,
            'ci95_low': mean - half_width,
            'ci95_high': mean + half_width,
            'preferred_direction': preferred_direction,
            'positive_better_pairs': positive_better,
            'negative_better_pairs': negative_better,
            'tied_pairs': ties,
            'sign_test_p': exact_two_sided_sign_test(
                positive_better, negative_better)
            if preferred_direction != 'descriptive' else math.nan,
            'experiment_seeds': ';'.join(experiment_seeds),
            'per_repeat_differences': ';'.join(str(value) for value in differences),
        })
    return output


def write_summary(path, summaries):
    if not summaries:
        raise ValueError('no evaluation rows found')
    with open(path, 'w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)


def write_paired_summary(path, rows):
    if not rows:
        return
    with open(path, 'w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    parser.add_argument('--positive-method', default='mappo')
    parser.add_argument('--negative-method', default='execution_aware')
    parser.add_argument('--paired-output', default='paired_differences.csv')
    args = parser.parse_args()
    rows = read_final_rows(args.directory)
    summaries = summarize(rows)
    output_path = os.path.join(args.directory, 'summary.csv')
    write_summary(output_path, summaries)
    paired_path = os.path.join(args.directory, args.paired_output)
    write_paired_summary(
        paired_path,
        paired_differences(rows, args.positive_method, args.negative_method))
    for row in summaries:
        print(
            '{method}: coverage={coverage_pct_mean:.2f}+/-{coverage_pct_std:.2f}, '
            'path={path_length_total_m_mean:.2f}+/-{path_length_total_m_std:.2f}, '
            'reached={goals_reached_mean:.1f}+/-{goals_reached_std:.1f}'.format(**row))
    print('summary: %s' % output_path)
    print('paired differences: %s' % paired_path)


if __name__ == '__main__':
    main()
