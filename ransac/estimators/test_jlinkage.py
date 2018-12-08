import unittest

import numpy as np

from ransac.estimators import jlinkage
from ransac.models import least_squares


class TestJLinkage(unittest.TestCase):

    def setUp(self):
        model = least_squares.LeastSquaresModel()
        num_sample_points = 2
        min_inliers = 0
        inlier_threshold = 1
        stop_iterations = 10
        self.jlink = jlinkage.JLinkage(
            model, num_sample_points, min_inliers,
            inlier_threshold, stop_iterations, random_seed=0)

    def test_one_cluster(self):
        data = np.array([
            (1, 0), (2, 0), (4, 0), (5, 0), (6, 0), (10, 0), (25, 0)])
        results = self.jlink.run(data)
        models = results.get_model_results()
        self.assertEqual(len(models), 1)
        self.assertEqual(set(map(tuple, models[0].inliers)), set(map(tuple, data)))

    def test_two_clusters(self):
        group_one = ((1, 0), (2, 0), (4, 0), (5, 0), (100, 0))
        group_two = ((-1, -50), (-3, -53), (-4, -54), (-5, -55))
        data = np.array(group_one + group_two)
        results = self.jlink.run(data)
        models = results.get_model_results()
        inlier_sets = [frozenset(map(tuple, m.inliers)) for m in models]
        self.assertEqual(len(models), 2)
        self.assertTrue(frozenset(group_one) in inlier_sets)
        self.assertTrue(frozenset(group_two) in inlier_sets)


class TestJDistTable(unittest.TestCase):

    def setUp(self):
        self.item_a = frozenset({'a'})
        self.item_b = frozenset({'b'})
        self.item_c = frozenset({'c'})
        self.item_d = frozenset({'d'})

    def test_happy(self):
        self.item_to_set_mask = {
            self.item_a: np.array([True, True, True, True, True]),
            self.item_b: np.array([False, False, False, False, False]),
            self.item_c: np.array([True, True, False, False, False]),
            self.item_d: np.array([True, True, True, False, False])}
        table = jlinkage.JDistTable(self.item_to_set_mask)
        pair, dist = table.get_nearest_pair()
        self.assertEqual(frozenset(pair), frozenset({self.item_c, self.item_d}))
        self.assertAlmostEqual(dist, 1 / float(3))

    def test_empty(self):
        table = jlinkage.JDistTable({})
        self.assertEqual(table.get_nearest_pair(), (None, None))

    def test_disjoint(self):
        item_to_set_mask = {
            self.item_a: np.array([True, True, True, True]),
            self.item_b: np.array([False, False, False, False])}
        table = jlinkage.JDistTable(item_to_set_mask)
        pair, dist = table.get_nearest_pair()
        self.assertEqual(frozenset(pair), frozenset({self.item_a, self.item_b}))
        self.assertAlmostEqual(dist, 1)

    def test_same_set(self):
        item_to_set_mask = {
            self.item_a: np.array([True, True, True, True]),
            self.item_b: np.array([True, True, True, True])}
        table = jlinkage.JDistTable(item_to_set_mask)
        pair, dist = table.get_nearest_pair()
        self.assertEqual(frozenset(pair), frozenset({self.item_a, self.item_b}))
        self.assertAlmostEqual(dist, 0)

    def test_add_item(self):
        table = jlinkage.JDistTable({})
        self.assertEqual(table.get_nearest_pair(), (None, None))
        table.add_item(self.item_a, np.array([True]))
        self.assertEqual(table.get_nearest_pair(), (None, None))
        table.add_item(self.item_b, np.array([False]))
        pair, dist = table.get_nearest_pair()
        self.assertEqual(frozenset(pair), frozenset({self.item_a, self.item_b}))
        self.assertAlmostEqual(dist, 1)
        table.add_item(self.item_c, np.array([True]))
        pair, dist = table.get_nearest_pair()
        self.assertEqual(frozenset(pair), frozenset({self.item_a, self.item_c}))
        self.assertAlmostEqual(dist, 0)
        self.assertEqual(
            table.get_all_items(),
            frozenset({self.item_a, self.item_b, self.item_c}))

    def test_remove_item(self):
        item_to_set_mask = {
            self.item_a: np.array([True]),
            self.item_b: np.array([False]),
            self.item_c: np.array([True])}
        table = jlinkage.JDistTable(item_to_set_mask)
        pair, dist = table.get_nearest_pair()
        self.assertEqual(frozenset(pair), frozenset({self.item_a, self.item_c}))
        self.assertAlmostEqual(dist, 0)
        table.remove_item(self.item_c)
        pair, dist = table.get_nearest_pair()
        self.assertEqual(frozenset(pair), frozenset({self.item_a, self.item_b}))
        self.assertAlmostEqual(dist, 1)
        self.assertEqual(
            table.get_all_items(),
            frozenset({self.item_a, self.item_b}))

    def test_jaccard_dist(self):
        self.assertEqual(
            jlinkage.JDistTable._jaccard_distance(
                np.array([True]), np.array([False])),
            1)
        self.assertEqual(
            jlinkage.JDistTable._jaccard_distance(
                np.array([True]), np.array([True])),
            0)
        self.assertEqual(
            jlinkage.JDistTable._jaccard_distance(
                np.array([True, True]), np.array([True, False])),
            0.5)
