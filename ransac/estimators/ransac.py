import math

import numpy as np

from ransac.models import exceptions


class Ransac(object):
    """Random Sample Consensus.

    This attempts to fit a model to some data, such that the fit is completely
    unaffected by outliers. The model is fitted against random samples of the
    data, repeatedly, before returning the model for the sample with lowest error.
    """

    def __init__(
            self, model, num_sample_points,
            min_inliers, inlier_threshold, stop_iterations, random_seed=None):
        """Constructor.

        Args:
            model: A Model subclass instance. This will be used
                to fit a mathematical model against the sampled data.
            num_sample_points: Integer, how many sample points to use per iteration.
            min_inliers: Integer, minimum number of inliers, excluding the
                initial sample points, for a model to be considered valid.
            inlier_threshold: Float, maximum size of the deviation of a data point
                from the model for it to still be considered an inlier.
            stop_iterations: Integer, how many RANSAC iterations to run. Higher
                iterations increase the chances of finding a better fit.
            random_seed: Optional random seed. Using a consistent integer seed will make
                output deterministic between runs.
        """
        self.model = model
        self.num_sample_points = num_sample_points
        self.min_inliers = min_inliers
        self.inlier_threshold = inlier_threshold
        self.stop_iterations = stop_iterations
        self.random = np.random.RandomState(seed=random_seed)

    def run(self, data):
        """Runs RANSAC.

        Args:
            data: Numpy array of data points.

        Returns:
            RansacHypothesis object with fit, inliers, and outliers attributes,
            or None if no valid model can be fit.
        """
        self._validate_args(data)

        best_model = None
        for model in self._generate_ransac_samples(data):
            if (best_model is None or
                    model.total_inlier_error < best_model.total_inlier_error):
                best_model = model

        return best_model

    def _validate_args(self, data):
        """Validates inputs.

        Raises:
            ValueError.
        """
        if len(data) < self.num_sample_points + self.min_inliers:
            raise ValueError('Too few data points to satisfy ' +
                             'num_sample_points and min_inliers.')

    def _generate_ransac_samples(self, data):
        """Generates valid RANSAC sample sets.

        Args:
            data: Numpy array of data points.

        Yields:
            RansacHypothesis instances.
        """
        for _ in range(self.stop_iterations):
            sample_mask = self._get_random_sample(data)
            try:
                model_fit = self.model.fit(data[sample_mask])
            except exceptions.DegenerateModelException:
                continue
            residuals, inlier_mask = self._apply_model(
                data, self.model, model_fit)
            inlier_mask |= sample_mask
            inlier_count = len(inlier_mask[inlier_mask])
            if inlier_count < self.min_inliers:
                continue
            # Acceptable model - refit it to the final inliers.
            model_fit = self.model.fit(data[inlier_mask])
            residuals, _ = self._apply_model(data, self.model, model_fit)
            yield RansacHypothesis(model_fit, data, residuals, inlier_mask)

    def _apply_model(self, data, model, model_fit):
        """Calculates residuals and inliers.

        Args:
            data: Numpy array of data points.
            model: Model subclass instance.
            model_fit: Numpy array of model fit parameters (each element is a separate model).

        Returns:
            Tuple of residuals array, inlier mask.
        """
        residuals = model.get_residuals(data, model_fit)
        inlier_mask = residuals <= self.inlier_threshold
        return residuals, inlier_mask

    def _get_random_sample(self, data):
        """Takes a random sample of our data.

        Args:
            data: Numpy array of data points.

        Returns:
            Sample mask.
        """
        sample_mask = np.zeros(data.shape[0], dtype=bool)
        sample_mask[list(random_sample(
            self.random, list(range(data.shape[0])), self.num_sample_points))] = True
        return sample_mask


class RansacHypothesis(object):
    """Stores a RANSAC model hypothesis."""

    def __init__(self, model_fit, data, residuals, inlier_mask):
        """Constructor.

        Args:
            model_fit: Any model fit parameters.
            data: Numpy array of data points.
            residuals: Numpy array of residuals. Order should match data.
            inlier_mask: Numpy mask which selects inliers from data or residuals.
        """
        self.fit = model_fit
        self.data = data
        self.residuals = residuals
        self._inlier_mask = inlier_mask
        # Per-point residuals.
        self._total_inlier_error_cached = None

    @property
    def inliers(self):
        return self.data[self._inlier_mask]

    @property
    def outliers(self):
        return self.data[~self._inlier_mask]

    @property
    def total_inlier_error(self):
        if self._total_inlier_error_cached is None:
            self._total_inlier_error_cached = np.sum(
                self.residuals[self._inlier_mask])
        return self._total_inlier_error_cached

    def __eq__(self, other):
        if isinstance(other, RansacHypothesis):
            return (np.array_equal(self.fit, other.fit) and
                    np.array_equal(self.data, other.data) and
                    np.array_equal(self.residuals, other.residuals) and
                    np.array_equal(self._inlier_mask, other._inlier_mask))
        return False

    def __hash__(self):
        return hash(tuple(sorted(self.__dict__.items())))


def calculate_ransac_iterations(min_points_per_sample, outlier_rate, desired_success_probability):
    """Estimates how many RANSAC iterations you should run.

    Args:
        min_points_per_sample: Minimum number of points to build a model hypothesis.
        outlier_rate: Float, 0-1, how often outliers are expected.
        desired_success_probability: Float, 0-1 exclusive, desired certainty that the
            RANSAC run will find the correct model. Higher certainty requires more iterations.

    Returns:
        Number of iterations.
    """
    if not 0 < desired_success_probability < 1:
        raise ValueError('desired_success_probability should fall between 0 and 1, exclusive.')
    if not 0 <= outlier_rate <= 1:
        raise ValueError('outlier_rate should fall between 0 and 1.')
    if min_points_per_sample <= 0:
        raise ValueError('min_points_per_sample must be a positive integer.')
    assert 0 < desired_success_probability < 1
    return math.ceil(math.log(1 - desired_success_probability) /
                     math.log((1 - (1 - outlier_rate) ** min_points_per_sample)))


def random_sample(random_state, population, sample_size):
    """Samples a given number of points from the population without replacement.

    Floyd's algorithm.

    Args:
        random_state: RandomState instance.
        population: An iterable from which to draw our sample.
        sample_size: Integer, How many elements to return. Must be <= len(population).

    Returns:
        Set with sample_size elements.
    """
    sample = set()
    pop_size = len(population)
    if pop_size < sample_size:
        raise ValueError('Sample size exceeds population size.')
    for i in range(pop_size - sample_size, pop_size):
        ind = random_state.randint(0, i + 1)
        if population[ind] not in sample:
            sample.add(population[ind])
        else:
            sample.add(population[i])
    return sample
