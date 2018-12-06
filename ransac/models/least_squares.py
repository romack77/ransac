import numpy as np

from ransac.models.base import Model


class LeastSquaresModel(Model):

    def fit(self, data):
        xs = data[:, 0]
        ys = data[:, 1]
        coeffs = np.vstack((xs, np.ones(len(data)))).T
        solution, residuals, _, _ = np.linalg.lstsq(coeffs, ys, rcond=None)
        return solution

    def predict(self, xs, model_fit):
        predicted_ys = model_fit[0] * xs + model_fit[1]
        return predicted_ys
