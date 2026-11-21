import json
import os
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(__file__)), 'scripts', 'voronoi_exploration'))

from frontier_decision_trace import FrontierDecisionTrace


class FrontierDecisionTraceTests(unittest.TestCase):
    def test_assignments_have_exactly_one_outcome_and_keep_censored_attempts(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'run.jsonl'
            trace = FrontierDecisionTrace(path, experiment_seed=501)
            trace.assign('robot1', 1.0, 2.0, selected_frontier_id=6)
            trace.assign('robot2', 2.0, 2.5, selected_frontier_id=9)
            trace.finish('robot1', 'reached', 4.0, 3.0)
            trace.finish('robot1', 'reached', 4.2, 3.0)
            trace.assign('robot2', 5.0, 3.5, selected_frontier_id=10)
            trace.close('simulation_horizon', 300.0, 4.0)
            trace.close('run_shutdown', 301.0, 4.0)
            records = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual(records[0]['experiment_seed'], 501)
        self.assertEqual(records[-1]['reason'], 'simulation_horizon')
        outcomes = [row for row in records if row['event'] == 'outcome']
        self.assertEqual([(row['attempt_id'], row['outcome']) for row in outcomes],
                         [(1, 'reached'), (2, 'replaced'), (3, 'simulation_horizon')])
        self.assertEqual([row['team_area_gain_m2'] for row in outcomes], [1.0, 1.0, 0.5])

    def test_existing_trace_is_preserved(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'run.jsonl'
            trace = FrontierDecisionTrace(path, experiment_seed=501)
            trace.close('run_shutdown', 1.0, 0.0)
            original = path.read_bytes()
            with self.assertRaises(FileExistsError):
                FrontierDecisionTrace(path, experiment_seed=502)
            self.assertEqual(path.read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
