#!/usr/bin/env python3
"""Small dependency-free tests for v1 experiment geometry helpers."""

import unittest

import vlm_action_geometry_v1_common as common


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.square = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]

    def test_convex_hull(self):
        self.assertEqual(len(common.convex_hull(self.square + [(0.0, 0.0)])), 4)

    def test_polygon_distance(self):
        shifted = common.translate_polygon(self.square, 5.0, 0.0)
        self.assertAlmostEqual(common.polygon_distance(self.square, shifted), 3.0)

    def test_polygon_intersection(self):
        shifted = common.translate_polygon(self.square, 1.0, 0.0)
        self.assertTrue(common.polygons_intersect(self.square, shifted))

    def test_rotation(self):
        rotated = common.rotate_polygon(self.square, 90.0)
        self.assertAlmostEqual(common.polygon_distance(self.square, rotated), 0.0)

    def test_labels(self):
        self.assertEqual(common.translation_level(5.0), "SHORT")
        self.assertEqual(common.translation_level(5.1), "MEDIUM")
        self.assertEqual(common.rotation_level(45.0), "MEDIUM")
        self.assertEqual(common.rotation_level(45.1), "LARGE")

    def test_sparse_ids(self):
        first = common.stable_sparse_ids("scene", ["a", "b", "c", "d"])
        second = common.stable_sparse_ids("scene", ["a", "b", "c", "d"])
        self.assertEqual(first, second)
        self.assertEqual(len(set(first[0])), 4)
        self.assertTrue(all(10 <= value <= 99 for value in first[0]))


if __name__ == "__main__":
    unittest.main()
