import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / \
         'summarize_policy_baseline_study.py'
SPEC = importlib.util.spec_from_file_location('policy_summary', SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PolicyBaselineSummaryTests(unittest.TestCase):
    def write_metrics(self, path, coverage, path_length, completion):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=[
                'coverage_pct', 'path_length_total_m', 'goals_reached'])
            writer.writeheader()
            writer.writerow({
                'coverage_pct': coverage - 1.0,
                'path_length_total_m': path_length - 1.0,
                'goals_reached': completion - 1.0,
            })
            writer.writerow({
                'coverage_pct': coverage,
                'path_length_total_m': path_length,
                'goals_reached': completion,
            })

    def test_loads_final_rows_and_aggregates_methods(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for training_seed in (1101, 2202):
                lineage = root / 'training_seed_{}'.format(training_seed)
                self.write_metrics(
                    lineage / 'repeat_01_ippo.csv', 40.0, 20.0, 0.4)
                self.write_metrics(
                    lineage / 'repeat_01_mappo.csv', 50.0, 18.0, 0.6)
            runs = MODULE.load_runs(root)
            self.assertEqual(len(runs), 4)
            self.assertEqual({run['training_seed'] for run in runs}, {1101, 2202})
            lineages = MODULE.lineage_means(runs)
            self.assertEqual(len(lineages), 4)
            summary = {row['method']: row for row in MODULE.summarize(lineages)}
            self.assertEqual(summary['mappo']['independent_training_seeds'], 2)
            self.assertAlmostEqual(summary['ippo']['coverage_pct_mean'], 40.0)
            self.assertAlmostEqual(summary['mappo']['coverage_pct_mean'], 50.0)
            self.assertAlmostEqual(summary['mappo']['path_length_total_m_mean'], 18.0)


if __name__ == '__main__':
    unittest.main()
