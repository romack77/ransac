import abc


class Model(metaclass=abc.ABCMeta):
    """Abstract base class for models."""

    @abc.abstractmethod
    def fit(self, data):
        """Fits a model to the given data.

        Args:
            data: A numpy array of data points.

        Returns:
            Arbitrary model parameters. These are
            passed as-is to get_residuals() and predict().

        Raises:
            DegenerateModelException, if model fitting fails.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, xs, fit):
        """Predicts results from a fitted model to the given data.

        Args:
            xs: A numpy array of input values.
            fit: The model parameters from model.fit().

        Returns:
            A numpy array of predicted values.
        """
        raise NotImplementedError

    def get_residuals(self, data, fit):
        """Calculates residuals between observed data and a fitted model.

        Args:
            data: A numpy array of observed data points.
            fit: The model parameters from model.fit().

        Returns:
            A numpy array of residuals, that is, the squared deviations of
            the observed values from the predicted values.
        """
        xs = data[:, 0]
        ys = data[:, 1]
        predicted_ys = self.predict(xs, fit)
        return (ys - predicted_ys) ** 2


