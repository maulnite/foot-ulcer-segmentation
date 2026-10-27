"""Analytic and independent-oracle tests; run in the analysis environment."""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from pfom_metrics import canny_edges, pratt_from_edges, score_masks


def edges(points, shape=(17, 17)):
    result = np.zeros(shape, dtype=bool)
    for point in points:
        result[point] = True
    return result


class TestPFOM(unittest.TestCase):
    def test_perfect_nonempty_edges_equal_one(self):
        edge = edges([(3, 4), (6, 8), (7, 8)])
        self.assertEqual(pratt_from_edges(edge, edge)["pfom"], 1)

    def test_known_distances_and_squared_euclidean_penalty(self):
        for point, expected in [((5, 6), .9), ((5, 8), .5), ((8, 9), 9 / 34)]:
            with self.subTest(point=point):
                result = pratt_from_edges(edges([(5, 5)]), edges([point]))
                self.assertAlmostEqual(result["pfom"], expected, places=14)

    def test_missing_edges_use_max_count_normalization(self):
        result = pratt_from_edges(edges([(3, 3), (3, 9)]), edges([(3, 3)]))
        self.assertEqual(result["pfom"], .5)

    def test_extra_edges_penalized_and_direction_is_explicit(self):
        one, two = edges([(3, 3)]), edges([(3, 3), (3, 8)])
        forward, reverse = pratt_from_edges(one, two), pratt_from_edges(two, one)
        self.assertAlmostEqual(forward["pfom"], (1 + 9 / 34) / 2)
        self.assertEqual(reverse["pfom"], .5)
        self.assertNotEqual(forward["pfom"], reverse["pfom"])

    def test_empty_edge_policies_are_separate(self):
        empty, nonempty = edges([]), edges([(5, 5)])
        for truth, pred, value, case in [(empty, empty, 1, "both_empty"),
                                        (empty, nonempty, 0, "gt_empty"),
                                        (nonempty, empty, 0, "prediction_empty")]:
            with self.subTest(case=case):
                result = pratt_from_edges(truth, pred)
                self.assertEqual(result["pfom"], value)
                self.assertEqual(result["edge_case"], case)
                self.assertEqual(result["pfom_paper_sentinel2_proxy"], 2)

    def test_distance_transform_matches_independent_brute_force(self):
        rng = np.random.default_rng(42)
        for _ in range(40):
            gt_points = np.unravel_index(rng.choice(17 * 17, size=int(rng.integers(1, 20)), replace=False), (17, 17))
            pred_points = np.unravel_index(rng.choice(17 * 17, size=int(rng.integers(1, 20)), replace=False), (17, 17))
            gt, pred = edges(list(zip(*gt_points))), edges(list(zip(*pred_points)))
            terms = []
            for p in zip(*pred_points):
                nearest_squared = min(sum((int(a) - int(b)) ** 2 for a, b in zip(p, g)) for g in zip(*gt_points))
                terms.append(1 / (1 + nearest_squared / 9))
            expected = sum(terms) / max(len(gt_points[0]), len(pred_points[0]))
            self.assertAlmostEqual(pratt_from_edges(gt, pred)["pfom"], expected, places=14)

    def test_canny_mask_identity_and_displacement(self):
        mask = np.zeros((41, 41), dtype=bool)
        mask[10:27, 10:27] = True
        self.assertGreater(canny_edges(mask).sum(), 0)
        self.assertEqual(score_masks(mask, mask)["pfom"], 1)
        one = score_masks(mask, np.roll(mask, 1, axis=1))["pfom"]
        three = score_masks(mask, np.roll(mask, 3, axis=1))["pfom"]
        self.assertTrue(0 < three < one < 1)
        self.assertEqual(score_masks(mask, np.zeros_like(mask))["pfom"], 0)

    def test_constant_masks_have_no_canny_edges(self):
        for value in [False, True]:
            mask = np.full((41, 41), value, dtype=bool)
            self.assertEqual(canny_edges(mask).sum(), 0)
            self.assertEqual(score_masks(mask, mask)["edge_case"], "both_empty")

    def test_reject_invalid_inputs_and_parameters(self):
        valid = edges([(5, 5)])
        for invalid in [np.zeros((2, 3, 1)), np.zeros((0, 2)), np.array([[0, .5]]), np.array([[np.nan]])]:
            with self.subTest(shape=invalid.shape), self.assertRaises(ValueError):
                pratt_from_edges(invalid, valid)
        with self.assertRaises(ValueError):
            pratt_from_edges(valid, np.zeros((3, 3)))
        for beta in [0, -1, np.inf, np.nan]:
            with self.subTest(beta=beta), self.assertRaises(ValueError):
                pratt_from_edges(valid, valid, beta=beta)
        with self.assertRaises(ValueError):
            canny_edges(valid, sigma=-1)
        with self.assertRaises(ValueError):
            canny_edges(valid, low_threshold=.2, high_threshold=.1)


if __name__ == "__main__":
    unittest.main()
