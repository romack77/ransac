import unittest

import numpy as np

from ransac.estimators import ransac
from ransac.models import base
from ransac.models import exceptions


class TestRansac(unittest.TestCase):

    class NoErrorModel(base.Model):

        def fit(self, data):
            return data.shape

        def predict(self, xs, model_fit):
            pass

        def get_residuals(self, data, model_fit):
            return np.zeros(data.shape[0])

    class OneOutlierModel(NoErrorModel):
        """Error=1 for the final data point, 0 for everything else."""

        def get_residuals(self, data, model_fit):
            residuals = np.zeros(data.shape[0])
            residuals[-1] = 1
            return residuals

    class DegenerateModel(NoErrorModel):

        def fit(self, data):
            raise exceptions.DegenerateModelException()

    def setUp(self):
        num_sample_points = 3
        min_inliers = 2
        inlier_threshold = 1
        stop_iterations = 3
        self.ransac = ransac.Ransac(
            TestRansac.NoErrorModel(), num_sample_points, min_inliers,
            inlier_threshold, stop_iterations, random_seed=0)

    def test_identity(self):
        data = np.array([(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)])
        result = self.ransac.run(data)
        inliers = result.inliers.tolist()
        inliers.sort()
        self.assertEqual(inliers, data.tolist())
        self.assertEqual(result.outliers.tolist(), [])
        self.assertEqual(result.total_inlier_error, 0)
        self.assertEqual(np.sum(result.residuals), 0)

    def test_outliers(self):
        num_sample_points = 2
        min_inliers = 0
        inlier_threshold = 0.5
        stop_iterations = 3
        my_ransac = ransac.Ransac(
            TestRansac.OneOutlierModel(), num_sample_points, min_inliers,
            inlier_threshold, stop_iterations, random_seed=0)
        data = np.array([(0, 0), (1, 1), (2, 2)])
        result = my_ransac.run(data)
        # OneOutlierModel gives one data point an error of 1, which is
        # beyond our outlier threshold.
        self.assertEqual(result.inliers.tolist(), [[0, 0], [1, 1]])
        self.assertEqual(result.outliers.tolist(), [[2, 2]])
        self.assertEqual(result.total_inlier_error, 0)
        self.assertEqual(np.sum(result.residuals), 1)

    def test_too_few_data_points(self):
        data = np.array([(0, 0), (1, 1), (2, 2), (3, 3)])
        with self.assertRaises(ValueError):
            self.ransac.run(data)

    def test_just_enough_data_points(self):
        data = np.array([(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)])
        self.assertIsNotNone(self.ransac.run(data))

    def test_degenerate_model(self):
        data = np.array([(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)])
        my_ransac = ransac.Ransac(
            TestRansac.DegenerateModel(), 2, 0, 0.5, 3, random_seed=0)
        results = my_ransac.run(data)
        self.assertIsNone(results)

    def test_calculate_ransac_iterations(self):
        self.assertEqual(ransac.calculate_ransac_iterations(2, .30, .95), 5)
        self.assertEqual(ransac.calculate_ransac_iterations(2, .30, .99), 7)
        self.assertEqual(ransac.calculate_ransac_iterations(2, .50, .95), 11)
        self.assertEqual(ransac.calculate_ransac_iterations(2, .80, .95), 74)
        self.assertEqual(ransac.calculate_ransac_iterations(4, .30, .95), 11)
        self.assertEqual(ransac.calculate_ransac_iterations(2, .95, .95), 1197)


class TestRandomSample(unittest.TestCase):

    def setUp(self):
        self.random = np.random.RandomState(0)

    def test_happy(self):
        sample = ransac.random_sample(self.random, [1, 2, 3], 1)
        self.assertEqual(sample, {1})
        sample = ransac.random_sample(self.random, [1, 2, 3], 2)
        self.assertEqual(sample, {2, 3})

    def test_whole_population(self):
        self.assertEqual(
            ransac.random_sample(self.random, [1, 2], 2),
            {1, 2})

    def test_empty_population(self):
        self.assertEqual(
            ransac.random_sample(self.random, [], 0),
            set([]))

    def test_zero_sample(self):
        self.assertEqual(
            ransac.random_sample(self.random, [1, 2], 0),
            set([]))

    def test_sample_too_large(self):
        with self.assertRaises(ValueError):
            ransac.random_sample(self.random, [], 1)
        with self.assertRaises(ValueError):
            ransac.random_sample(self.random, [1], 2)
