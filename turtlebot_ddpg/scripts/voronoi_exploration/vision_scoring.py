"""Lightweight visual openness scores for Gazebo exploration cameras."""

import cv2
import numpy as np


VISION_SCORING_MODES = (
    'projected_obstacle', 'projected_obstacle_gated', 'canny')


def _thirds(array):
    width = array.shape[1]
    third = max(1, width // 3)
    return (array[:, :third], array[:, third:2 * third], array[:, 2 * third:])


def canny_openness(img_rgb):
    """Legacy baseline: inverse Canny edge density in image thirds."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return tuple(1.0 - float(region.mean()) / 255.0
                 for region in _thirds(edges))


def projected_obstacle_openness(img_rgb):
    """Estimate free-looking directions from foreground projection area.

    Gazebo's camera view has a nearly uniform sky above the horizon and a
    nearly uniform floor below it. Their colours are estimated from each
    frame, so the score does not depend on a particular obstacle colour.
    Pixels unlike both references are treated as projected obstacle area.
    A large nearby obstacle therefore receives a lower openness score than
    a narrow, distant wall or passage.
    """
    if img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
        raise ValueError('expected an HxWx3 RGB image')
    height, width = img_rgb.shape[:2]
    if height < 8 or width < 3:
        raise ValueError('image is too small to score')

    image = img_rgb.astype(np.float32)
    x0, x1 = int(0.1 * width), max(int(0.9 * width), 1)
    sky = np.median(image[:max(int(0.12 * height), 1), x0:x1], axis=(0, 1))
    floor = np.median(
        image[int(0.82 * height):max(int(0.96 * height), 1), x0:x1],
        axis=(0, 1))

    sky_distance = np.linalg.norm(image - sky, axis=2)
    floor_distance = np.linalg.norm(image - floor, axis=2)
    background_distance = np.minimum(sky_distance, floor_distance)

    # A soft transition avoids a brittle colour threshold. Values below
    # eight intensity units are background-like; by 68 units a pixel is a
    # full foreground contribution.
    foreground = np.clip((background_distance - 8.0) / 60.0, 0.0, 1.0)
    return tuple(1.0 - float(region.mean()) for region in _thirds(foreground))


def projected_obstacle_confidence(img_rgb):
    """Return a conservative confidence for the adaptive colour model.

    The projected-obstacle score assumes that the upper and lower reference
    strips provide distinguishable sky and floor samples. Very dark, nearly
    uniform, or over-exposed images violate that assumption. The confidence
    gate suppresses the visual feature in those cases instead of injecting an
    arbitrary directional preference into the high-level policy.
    """
    if img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
        raise ValueError('expected an HxWx3 RGB image')
    height, width = img_rgb.shape[:2]
    if height < 8 or width < 3:
        raise ValueError('image is too small to score')

    image = img_rgb.astype(np.float32)
    x0, x1 = int(0.1 * width), max(int(0.9 * width), 1)
    sky = np.median(image[:max(int(0.12 * height), 1), x0:x1], axis=(0, 1))
    floor = np.median(
        image[int(0.82 * height):max(int(0.96 * height), 1), x0:x1],
        axis=(0, 1))
    reference_separation = float(np.linalg.norm(sky - floor))
    intensity_span = float(np.percentile(image, 95) - np.percentile(image, 5))
    reference_confidence = np.clip(
        (reference_separation - 4.0) / 16.0, 0.0, 1.0)
    exposure_confidence = np.clip(
        (intensity_span - 8.0) / 24.0, 0.0, 1.0)
    return float(min(reference_confidence, exposure_confidence))


def confidence_gated_projected_obstacle_openness(img_rgb):
    """Projected-obstacle openness with unreliable frames neutralised."""
    scores = projected_obstacle_openness(img_rgb)
    confidence = projected_obstacle_confidence(img_rgb)
    return tuple(confidence * value for value in scores)


def openness_from_image(img_rgb, mode='projected_obstacle'):
    if mode == 'projected_obstacle':
        return projected_obstacle_openness(img_rgb)
    if mode == 'projected_obstacle_gated':
        return confidence_gated_projected_obstacle_openness(img_rgb)
    if mode == 'canny':
        return canny_openness(img_rgb)
    raise ValueError('unknown vision scoring mode: %s' % mode)
