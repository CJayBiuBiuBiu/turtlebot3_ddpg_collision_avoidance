#!/usr/bin/env python3

import os
import sys
import unittest

import numpy as np


SCRIPT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), 'scripts', 'voronoi_exploration')
sys.path.insert(0, SCRIPT_DIR)

from vision_scoring import openness_from_image, projected_obstacle_openness


def synthetic_camera_frame(obstacle_colour):
    image = np.empty((120, 300, 3), dtype=np.uint8)
    image[:60] = 178
    image[60:] = 155
    image[25:85, 0:100] = obstacle_colour
    image[48:68, 100:200] = obstacle_colour
    return image


class VisionScoringTests(unittest.TestCase):
    def test_projected_area_orders_near_obstacle_narrow_wall_and_open_view(self):
        left, centre, right = projected_obstacle_openness(
            synthetic_camera_frame((120, 70, 45)))
        self.assertLess(left, centre)
        self.assertLess(centre, right)

    def test_score_does_not_depend_on_one_obstacle_colour(self):
        brown = projected_obstacle_openness(
            synthetic_camera_frame((120, 70, 45)))
        blue = projected_obstacle_openness(
            synthetic_camera_frame((50, 100, 160)))
        self.assertEqual(int(np.argmax(brown)), 2)
        self.assertEqual(int(np.argmax(blue)), 2)

    def test_moderate_global_brightness_shift_preserves_direction_order(self):
        image = synthetic_camera_frame((120, 70, 45))
        brighter = np.clip(image.astype(np.int16) + 20, 0, 255).astype(np.uint8)
        original_scores = projected_obstacle_openness(image)
        brighter_scores = projected_obstacle_openness(brighter)
        self.assertEqual(np.argsort(original_scores).tolist(),
                         np.argsort(brighter_scores).tolist())

    def test_mild_sensor_noise_preserves_openest_direction(self):
        rng = np.random.RandomState(7)
        image = synthetic_camera_frame((120, 70, 45)).astype(np.int16)
        noisy = np.clip(image + rng.normal(0.0, 3.0, image.shape),
                        0, 255).astype(np.uint8)
        scores = projected_obstacle_openness(noisy)
        self.assertEqual(int(np.argmax(scores)), 2)

    def test_unknown_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            openness_from_image(synthetic_camera_frame((120, 70, 45)), 'unknown')


if __name__ == '__main__':
    unittest.main()
