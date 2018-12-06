import unittest

import numpy as np

from ransac.models import least_squares


class TestLeastSquaresModel(unittest.TestCase):

    def test_happy(self):
        model = least_squares.LeastSquaresModel()
        xs = np.arange(4)
        ys = np.array([10, 2, 3, 4])
        data = np.column_stack((xs, ys))
        fit = model.fit(data)
        self.assertAlmostEqual(fit[0], -1.7)
        self.assertAlmostEqual(fit[1], 7.3)
        residuals = model.get_residuals(data, fit)
        self.assertEqual(tuple(int(r) for r in residuals), (7, 12, 0, 3))
