import math

import numpy as np
from scipy import optimize
from scipy import signal
from scipy import special

from ransac.estimators import ransac


class XRansac(ransac.Ransac):
    """A RANSAC variant that can find multiple models in the underlying data.

    This is largely the classic RANSAC algorithm, except that instead of
    picking a "best" model, it analyzes all the generated models to
    estimate the true number of models, and returns that many fitted models.

    More specifically:
    * Instead of storing only the best generated model, we store all of them.
    * We build a histogram of the residuals of each data point across all models.
    * The average number of peaks across all histograms estimates the number of models.
        This is because clustered data points will have similar residuals to
        an average randomly-generated model.
    * We select a "best" model for each of the peaks.
        1) We take the histogram of the point with the strongest peaks, and take the
            models making up those peaks as our candidates.
        2) We build a histogram for another point, limited to the candidate models, and
            retain only those candidates that remained clustered. This filters out spurious
            models that only happened to have a similar residual for the initial point.
        3) Of the final candidates, we select the center of each peak (i.e. median residual).
    """

    def __init__(
            self, model, num_sample_points,
            min_inliers, inlier_threshold, stop_iterations, random_seed=None,
            residual_histogram_num_bins=100, min_prominence=5):
        """Constructor. See base class.

        Additional Args:
            residual_histogram_num_bins: Integer, number of bins to use for
                residual histograms. These are searched for peaks to determine
                the number of models. This should be small enough to form peaks,
                but large enough to keep individual peaks distinct.
            min_prominence: Float, only find peaks in the residual histograms with
                this much vertical distance from their highest contour line.
        """
        self.residual_histogram_num_bins = residual_histogram_num_bins
        self.min_prominence = min_prominence
        super().__init__(
            model, num_sample_points, min_inliers,
            inlier_threshold, stop_iterations, random_seed=random_seed)

    def run(self, data):
        """Runs X-RANSAC.

        Args:
            data: Numpy array of data points.

        Returns:
            MultiRansacResult instance with get_model_results() and
            get_global_outliers() methods.
        """
        self._validate_args(data)

        # This essentially runs RANSAC without picking a winner.
        models = list(self._generate_ransac_samples(data))

        # Use all generated models to determine the number of underlying models.
        num_models, residual_histograms = self._estimate_number_of_models(
            models)
        if num_models < 1:
            return None

        best_models = self._select_n_best_models(num_models, residual_histograms)

        # Format the results.
        result = MultiRansacResult()
        all_inliers = []
        all_outliers = []
        for model in best_models:
            result.add_result(model)
            all_inliers.extend(model.inliers)
            all_outliers.extend(model.outliers)
        result.set_global_outliers(np.array(list({
            tuple(o) for o in all_outliers}.difference({
                tuple(i) for i in all_inliers}))))
        return result

    def _build_residual_histograms(self, models, residuals, bins):
        """Builds per-data point histograms of residuals under all models.

        Args:
            models: List of RansacHypothesis instances.
            residuals: List of residuals for each data point. Each entry is
                a list of residuals for that data point under each model.
            bins: Integer number of histogram bins to create.

        Returns:
            List of ResidualHistogram instances, one per data point.
        """
        residual_histograms = []
        for point_residuals in residuals:
            point_residuals = np.array(point_residuals)
            counts, bins = np.histogram(point_residuals, bins=bins)
            residual_histograms.append(ResidualHistogram(
                counts, bins, point_residuals, models, self.min_prominence))
        return residual_histograms

    def _estimate_number_of_models(self, models):
        """Estimates the number of models in the underlying data.

        Args:
            models: List of RansacHypothesis objects.

        Returns:
            Tuple of:
                Estimated number of underlying models. Can be 0.
                List of per-point ResidualHistogram instances.
        """
        residual_histograms = self._build_residual_histograms(
            models,
            list(zip(*[m.residuals for m in models])),
            bins=self.residual_histogram_num_bins)
        num_peaks_list = [
            len(h.get_peaks()[0]) for h in residual_histograms
            if len(h.get_peaks()[0]) != 0]
        num_models = int(np.median(num_peaks_list)) if num_peaks_list else 0
        return num_models, residual_histograms

    def _find_most_predictive_point(self, num_models, residual_histograms):
        """Finds the point with the strongest peaks, and the right number of peaks.

        Args:
            num_models: Integer, the estimated number of models.
            residual_histograms: List of ResidualHistogram instances, one for each data point.

        Returns:
            ResidualHistogram for the best point/best peaks.
        """
        best_histogram = None
        best_histogram_height = 0
        for i, histogram in enumerate(residual_histograms):
            peak_idxs, peak_heights = histogram.get_peaks()
            if len(peak_idxs) != num_models:
                continue
            total_height = np.sum(peak_heights)
            if total_height >= best_histogram_height:
                best_histogram_height = total_height
                best_histogram = histogram
        return best_histogram

    def _select_n_best_models(self, num_models, residual_histograms):
        """Finds the n best models among the RANSAC samples.

        Args:
            num_models: Estimated number of models.
            residual_histograms: List ResidualHistogram instances, one for each data point.

        Returns:
            List of n models.
        """
        assert num_models > 0

        # Find the point that most strongly predicted the correct number of models.
        best_histogram = self._find_most_predictive_point(num_models, residual_histograms)

        # With that point, find all the model hypotheses that contributed to its prediction.
        peak_models = best_histogram.get_models_in_peaks()

        # Some of the peak models may be spurious (incorrect models that just happened to have a
        # residual similar to the correct models). Take another random point to disambiguate them,
        # since they won't be clustered with the correct ones in the residual histogram
        # of another point.
        other_histogram = self.random.choice([
            h for h in residual_histograms if h != best_histogram], size=1)[0]
        best_models = []
        for models in peak_models:
            # This is finding the peak of the other point's residual histogram, but limited
            # to the initial point's peak models.
            peak_model_residuals = [
                r for (i, r) in enumerate(other_histogram.residuals)
                if other_histogram.models[i] in models]
            # Scale the number of bins according to the size of our subset of models,
            # otherwise peaks become harder to detect for smaller peak model sets.
            num_bins = math.ceil(
                self.residual_histogram_num_bins * (
                        len(peak_models) / len(best_histogram.models)))
            histogram = self._build_residual_histograms(
                models, [peak_model_residuals], num_bins)[0]
            center_models = histogram.get_models_in_peak_centers()
            if center_models:
                best_models.append(center_models[0])
            else:
                # If we couldn't find a peak, just take the model with median residual.
                best_models.append(histogram.get_model_with_median_residual(histogram.models))
        return best_models


class ResidualHistogram(object):
    """Stores residual histogram data for a data point."""

    def __init__(self, counts, bins, residuals, models, min_prominence):
        """Constructor.

        Args:
            counts: Array of histogram counts.
            bins: Array of histogram bin edges.
            residuals: Array of this point's residuals under each model.
            models: Array of models.
            min_prominence: Float, only find peaks with this much vertical distance
                from their lowest contour line.
        """
        self.counts = counts
        self.bins = bins
        # Per-model residuals
        self.residuals = residuals
        self.models = models
        self.min_prominence = min_prominence

        self._peaks_cached = None
        self._peak_models_cached = None
        self._peak_center_models_cached = None

    def get_peaks(self):
        """Finds peaks in the histogram.

        Returns:
             Tuple with list of histogram indices corresponding to peaks,
             and list of peak heights.
        """
        if self._peaks_cached is None:
            # find_peaks won't detect peaks without two neighbors (i.e. it will never
            # find a peak on the border), so give our histogram some dummy 0 zero edges.
            bordered_counts = np.concatenate((np.zeros(1), self.counts, np.zeros(1)))
            peaks, peak_data = signal.find_peaks(
                bordered_counts, height=(None, None), threshold=self.min_prominence)
            self._peaks_cached = ([p - 1 for p in peaks], list(peak_data['peak_heights']))
        return self._peaks_cached

    def get_models_in_peaks(self):
        """Extracts the models that formed peaks in the histogram.

        Returns:
            A per-peak list, each a list of models that belonged to the peak.
        """
        if self._peak_models_cached is None:
            # These are the hypotheses whose residuals for this point fell
            # into a bin on the point's residual histogram corresponding to a peak.
            peaks = self.get_peaks()[0]
            peak_models = [[] for _ in range(len(peaks))]
            for i, residual in enumerate(self.residuals):
                for j, peak in enumerate(peaks):
                    if len(self.bins) > peak and residual > self.bins[peak + 1]:
                        continue
                    if residual < self.bins[peak]:
                        continue
                    peak_models[j].append(self.models[i])
                    break
            self._peak_models_cached = peak_models
        return self._peak_models_cached

    def get_models_in_peak_centers(self):
        """Extracts the center model of each peak in the histogram.

        Returns:
            A list with one model per peak, each taken from the center of that peak.
        """
        if self._peak_center_models_cached is None:
            peak_center_models = []
            for peak_models in self.get_models_in_peaks():
                # Take the model at the center of the peak (median residual).
                center_model = self.get_model_with_median_residual(peak_models)
                peak_center_models.append(center_model)
            self._peak_center_models_cached = peak_center_models
        return self._peak_center_models_cached

    def get_model_with_median_residual(self, models):
        """Finds the model with median residual for this data point.

        Args:
            models: Any subset of my models.

        Returns:
            Model.
        """
        model_residuals = [
            r for (i, r) in enumerate(self.residuals)
            if self.models[i] in models]
        best_model_idx = np.argsort(model_residuals)[len(model_residuals) // 2]
        return models[best_model_idx]

    def __eq__(self, other):
        if isinstance(other, ResidualHistogram):
            return (np.array_equal(self.bins, other.bins) and
                    np.array_equal(self.counts, other.counts) and
                    np.array_equal(self.residuals, other.residuals) and
                    np.array_equal(self.models, other.models))
        return False

    def __hash__(self):
        return hash((self.bins, self.counts, self.residuals, self.models))


class MultiRansacResult(object):
    """Result class for estimators that may fit multiple models."""

    def __init__(self):
        self.results = []
        self.global_outliers = []

    def add_result(self, ransac_hypothesis):
        self.results.append(ransac_hypothesis)

    def set_global_outliers(self, outliers):
        self.global_outliers = outliers

    def get_model_results(self):
        """Returns a list of RansacHypothesis instances for each final model."""
        return self.results

    def get_global_outliers(self):
        """Returns data points that were not assigned to any model."""
        return self.global_outliers


def calculate_xransac_iterations(
        min_points_per_sample, outlier_rate, desired_success_probability,
        min_peak_samples=50, max_iterations=100000):
    """Estimates how many XRANSAC iterations you should run.

    Args:
        min_points_per_sample: Minimum number of points to build a model hypothesis.
        outlier_rate: Float, 0-1, how often outliers are expected.
        desired_success_probability: Float, 0-1 exclusive, desired certainty that the
            RANSAC run will find the correct model. Higher certainty requires more iterations.
        min_peak_samples: Integer, how many outlier free samples are needed to form peaks in
            the residual histogram.
        max_iterations: Integer, an upper bound on the true number of iterations. Optimization
            will fail if this is lower than the true value.

    Returns:
        Number of iterations.
    """
    if not 0 < desired_success_probability < 1:
        raise ValueError('desired_success_probability should fall between 0 and 1, exclusive.')
    if not 0 <= outlier_rate <= 1:
        raise ValueError('outlier_rate should fall between 0 and 1.')
    if min_points_per_sample <= 0:
        raise ValueError('min_points_per_sample must be a positive integer.')
    if min_peak_samples <= 0:
        raise ValueError('min_peak_samples must be a positive integer.')

    def success_margin(num_iterations):
        """Returns probability to succeed, minus desired_success_probability, thus optimal at 0."""
        total = 0
        for i in range(min_peak_samples):
            total += special.comb(num_iterations, i) * (
                    np.power(
                        1 - outlier_rate,
                        min_points_per_sample * i
                    ) *
                    np.power(
                        1 - np.power((1 - outlier_rate), min_points_per_sample),
                        num_iterations - i)
                    )
        return (1 - total) - desired_success_probability

    # Find the number of iterations that optimizes the desired success probability.
    return math.ceil(optimize.brentq(success_margin, 0, max_iterations))
