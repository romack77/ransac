import unittest

import numpy as np

from ransac.estimators import xransac
from ransac.estimators.test_ransac import TestRansac


class TestXRansac(unittest.TestCase):

    class AscendingErrorModel(TestRansac.NoErrorModel):
        """The index of a point is its error."""

        def get_residuals(self, data, model_fit):
            return np.array(range(len(data)))

    def test_happy(self):
        model = TestXRansac.AscendingErrorModel()
        my_ransac = xransac.XRansac(
            model, num_sample_points=2, min_inliers=0,
            inlier_threshold=5, stop_iterations=100,
            residual_histogram_num_bins=10)
        data = np.array([
            (0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (100, 100)])
        results = my_ransac.run(data)
        self.assertIsNotNone(results)
        self.assertEqual(len(results.get_model_results()), 1)

    def test_no_peaks(self):
        model = TestXRansac.AscendingErrorModel()
        my_ransac = xransac.XRansac(
            model, num_sample_points=2, min_inliers=0,
            inlier_threshold=5, stop_iterations=100,
            residual_histogram_num_bins=10, min_prominence=1000)
        data = np.array([
            (0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (100, 100)])
        results = my_ransac.run(data)
        self.assertIsNone(results)

    def test_calculate_xransac_iterations(self):
        self.assertEqual(
            xransac.calculate_xransac_iterations(4, .6, .99, min_peak_samples=50),
            2643)
        self.assertEqual(
            xransac.calculate_xransac_iterations(2, .30, .95, min_peak_samples=1),
            5)
        self.assertEqual(
            xransac.calculate_xransac_iterations(2, .30, .95, min_peak_samples=50),
            120)
        self.assertEqual(
            xransac.calculate_xransac_iterations(2, .30, .95, min_peak_samples=500),
            0)
        self.assertEqual(
            xransac.calculate_xransac_iterations(10, .30, .95, min_peak_samples=50),
            2195)
        self.assertEqual(
            xransac.calculate_xransac_iterations(2, .30, .99, min_peak_samples=50),
            128)
        self.assertEqual(
            xransac.calculate_xransac_iterations(2, .95, .90, min_peak_samples=50),
            23695)
        with self.assertRaises(ValueError):
            xransac.calculate_xransac_iterations(
                10, .30, .95, min_peak_samples=50, max_iterations=1)


class TestResidualHistogram(unittest.TestCase):

    def setUp(self):
        self.models = list(range(100))
        self.histo_empty = xransac.ResidualHistogram(
            [0], [10], [], [], 1)
        self.histo_one_peak = xransac.ResidualHistogram(
            [0, 2, 0], [0, 10, 20, 30], [11, 12], self.models[:2], 1)
        self.histo_edge_peak = xransac.ResidualHistogram(
            [2, 0, 0], [0, 10, 20, 30], [1, 2], self.models[:2], 1)
        self.histo_unprominent_peak = xransac.ResidualHistogram(
            [0, 2, 0], [0, 10, 20, 30], [11, 12], self.models[:2], 3)
        self.histo_three_peaks = xransac.ResidualHistogram(
            [2, 0, 2, 0, 3], [0, 10, 20, 30, 40, 50],
            [1, 1, 21, 21, 41, 41, 45], self.models[:7], 1)
        self.histo_adjacent_equal_peaks = xransac.ResidualHistogram(
            [2, 2, 0], [0, 10, 20, 30], [1, 1, 11, 11], self.models[:4], 1)
        self.histo_adjacent_stepped_peaks = xransac.ResidualHistogram(
            [2, 4, 0], [0, 10, 20, 30], [1, 1, 11, 11, 11, 11],
            self.models[:6], 1)

    def test_get_peaks(self):
        self.assertEqual(self.histo_empty.get_peaks(), ([], []))
        self.assertEqual(self.histo_one_peak.get_peaks(), ([1], [2]))
        self.assertEqual(self.histo_edge_peak.get_peaks(), ([0], [2]))
        self.assertEqual(self.histo_unprominent_peak.get_peaks(), ([], []))
        self.assertEqual(
            self.histo_three_peaks.get_peaks(),
            ([0, 2, 4], [2, 2, 3]))
        self.assertEqual(self.histo_adjacent_equal_peaks.get_peaks(), ([], []))
        self.assertEqual(self.histo_adjacent_stepped_peaks.get_peaks(), ([1], [4]))

    def test_get_models_in_peaks(self):
        self.assertEqual(
            self.histo_empty.get_models_in_peaks(), [])
        self.assertEqual(
            self.histo_one_peak.get_models_in_peaks(), [self.models[:2]])
        self.assertEqual(
            self.histo_edge_peak.get_models_in_peaks(), [self.models[:2]])
        self.assertEqual(
            self.histo_unprominent_peak.get_models_in_peaks(), [])
        self.assertEqual(
            self.histo_three_peaks.get_models_in_peaks(),
            [self.models[:2], self.models[2:4], self.models[4:7]])
        self.assertEqual(
            self.histo_adjacent_equal_peaks.get_models_in_peaks(), [])
        self.assertEqual(
            self.histo_adjacent_stepped_peaks.get_models_in_peaks(), [self.models[2:6]])

    def test_get_models_in_peak_centers(self):
        self.assertEqual(
            self.histo_empty.get_models_in_peak_centers(), [])
        self.assertEqual(
            self.histo_one_peak.get_models_in_peak_centers(), [self.models[1]])
        self.assertEqual(
            self.histo_edge_peak.get_models_in_peak_centers(), [self.models[1]])
        self.assertEqual(
            self.histo_unprominent_peak.get_models_in_peak_centers(), [])
        self.assertEqual(
            self.histo_three_peaks.get_models_in_peak_centers(),
            [self.models[1], self.models[3], self.models[5]])
        self.assertEqual(
            self.histo_adjacent_equal_peaks.get_models_in_peak_centers(), [])
        self.assertEqual(
            self.histo_adjacent_stepped_peaks.get_models_in_peak_centers(),
            [self.models[4]])

    def test_equality(self):
        same_peaks = [
            xransac.ResidualHistogram(
                [0, 2, 0], [0, 10, 20, 30], [11, 12], self.models[:2], 1)
            for _ in range(2)]
        diff_peak = xransac.ResidualHistogram(
            [0, 2, 0], [0, 10, 20, 30], [12, 13], self.models[:2], 1)
        self.assertEqual(same_peaks[0], same_peaks[1])
        self.assertEqual(same_peaks[1], same_peaks[0])
        self.assertNotEqual(same_peaks[0], diff_peak)
        self.assertNotEqual(self.histo_one_peak, self.histo_adjacent_equal_peaks)
