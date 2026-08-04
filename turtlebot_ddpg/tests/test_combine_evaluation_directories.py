#!/usr/bin/env python3

import os
import sys
import unittest
from unittest import mock


SCRIPTS_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', 'scripts'))
sys.path.insert(0, SCRIPTS_DIR)

import combine_evaluation_directories as combine


class CombineEvaluationDirectoriesTests(unittest.TestCase):
    def test_combined_repeat_ids_follow_unique_seed_order(self):
        rows_by_directory = {
            'old': [
                {'experiment_seed': '9200', 'selection_mode': 'geometric',
                 '_file': 'repeat_01_geometric.csv'},
                {'experiment_seed': '9200', 'selection_mode': 'mappo_tail',
                 '_file': 'repeat_01_mappo_tail.csv'},
            ],
            'new': [
                {'experiment_seed': '9205', 'selection_mode': 'geometric',
                 '_file': 'repeat_01_geometric.csv'},
                {'experiment_seed': '9205', 'selection_mode': 'mappo_tail',
                 '_file': 'repeat_01_mappo_tail.csv'},
            ],
        }
        with mock.patch.object(
                combine, 'read_final_rows', side_effect=rows_by_directory.get):
            rows, provenance = combine.combine_rows(('old', 'new'))
        self.assertEqual(len(provenance), 4)
        filenames = {row['_file'] for row in rows}
        self.assertIn('repeat_01_geometric.csv', filenames)
        self.assertIn('repeat_02_geometric.csv', filenames)

    def test_duplicate_seed_method_is_rejected(self):
        duplicate = [{
            'experiment_seed': '9200', 'selection_mode': 'geometric',
            '_file': 'repeat_01_geometric.csv',
        }]
        with mock.patch.object(
                combine, 'read_final_rows', return_value=duplicate):
            with self.assertRaisesRegex(ValueError, 'duplicate seed/method'):
                combine.combine_rows(('first', 'second'))


if __name__ == '__main__':
    unittest.main()
