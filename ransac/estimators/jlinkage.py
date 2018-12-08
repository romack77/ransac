import heapq
import itertools

import numpy as np

from ransac.estimators import ransac
from ransac.estimators import xransac
from ransac.models import exceptions


class JLinkage(ransac.Ransac):
    """Runs J Linkage clustering over RANSAC hypotheses.

    This can fit multiple models, and the number of models does
    not need to be specified in advance.

    J-linkage uses agglomerative clustering based on Jaccard distance
    (a measure of Set distance). Data points are grouped based on
    how many model hypotheses they shared inlier status in.

    A final model fitting is performed on the final clusters.
    """

    def run(self, data):
        """Runs J-Linkage.

        Args:
            data: Numpy array of data points.

        Returns:
            MultiRansacResult instance with get_model_results() and
            get_global_outliers() methods.
        """
        self._validate_args(data)

        # This essentially runs RANSAC without picking a winner.
        models = list(self._generate_ransac_samples(data))

        # Agglomerative clustering by Jaccard distance.
        cluster_to_preference_set = self._initialize_clusters(data, models)
        jdist_table = JDistTable(cluster_to_preference_set)
        while True:
            pair, jdist = jdist_table.get_nearest_pair()
            if jdist is None or jdist >= 1:
                # All clusters now have disjoint preference sets.
                break
            self._merge_clusters(
                cluster_to_preference_set, jdist_table, pair[0], pair[1])

        # Fit models for final clusters.
        point_clusters = cluster_to_preference_set.keys()
        all_inliers = []
        all_outliers = []
        result = xransac.MultiRansacResult()
        for point_cluster in point_clusters:
            try:
                model_fit = self.model.fit(np.array(point_cluster))
            except exceptions.DegenerateModelException:
                continue
            residuals = self.model.get_residuals(data, model_fit)
            model = ransac.RansacHypothesis(
                model_fit,
                data,
                residuals,
                np.array([
                    tuple(p) in point_cluster and residuals[i] <= self.inlier_threshold
                    for (i, p) in enumerate(data)]))
            if len(model.inliers) >= self.min_inliers:
                result.add_result(model)
                all_inliers.extend(model.inliers)
                all_outliers.extend(model.outliers)

        result.set_global_outliers(np.array(list({
            tuple(o) for o in all_outliers}.difference({
                tuple(i) for i in all_inliers}))))
        return result

    def _initialize_clusters(self, data, models):
        """Builds initial clusters for j-linkage.

        Args:
            data: Numpy array of data points.
            models: List of RansacHypothesis instances.

        Returns:
            Dict of cluster to preference set. A cluster is a
                set of data points. A preference set is a mask
                for models which selects models preferred by that
                cluster.
        """
        cluster_to_preference_set = {}
        for p in data:
            p = tuple(p)
            cluster_to_preference_set[(p,)] = np.array([
                p in m.inliers for m in models])
        return cluster_to_preference_set

    def _merge_clusters(self, cluster_to_preference_set, jdist_table, cluster_a, cluster_b):
        """Merges two clusters in the cluster table.

        Args:
            cluster_to_preference_set: Dict of cluster set to preference set array.
            cluster_a: Set, any key from the table.
            cluster_b: Set, another key from the table.
        """
        preference_set_a = cluster_to_preference_set[cluster_a]
        del cluster_to_preference_set[cluster_a]
        preference_set_b = cluster_to_preference_set[cluster_b]
        del cluster_to_preference_set[cluster_b]
        jdist_table.remove_item(cluster_a)
        jdist_table.remove_item(cluster_b)

        if cluster_a < cluster_b:
            new_cluster = cluster_a + cluster_b
        else:
            new_cluster = cluster_b + cluster_a
        new_preference_set = preference_set_a & preference_set_b
        cluster_to_preference_set[new_cluster] = new_preference_set
        jdist_table.add_item(new_cluster, new_preference_set)


class JDistTable(object):
    """Associates items by Jaccard distance."""

    def __init__(self, item_to_set_mask):
        """"Constructor.

        Args:
            item_to_set_mask: Dict of item to set mask. The set masks
                determine an item's distance to other items.
        """
        self._item_to_set_mask = dict(item_to_set_mask)
        # Maps pairs of items to Jaccard distance based on associated set masks.
        self._jdist_table = self._init_j_dist_table(item_to_set_mask)

    def add_item(self, item, set_mask):
        """Adds an item to the table."""
        for other_item, other_set_mask in self._item_to_set_mask.items():
            if item == other_item:
                continue
            j_dist = self._jaccard_distance(set_mask, other_set_mask)
            heapq.heappush(
                self._jdist_table, (j_dist, frozenset({item, other_item})))
        self._item_to_set_mask[item] = set_mask

    def remove_item(self, item):
        """Removes an item from the table."""
        del self._item_to_set_mask[item]

    def get_nearest_pair(self):
        """Finds the nearest pair of items.

        Currently this is an exhaustive scan of the pairwise lookup table.

        Returns:
            Tuple of (tuple - nearest item  pair, float, 0-1 - Jaccard distance).
            dist 0 means identical sets, dist 1 means disjoint sets.
            Both may be None if the table has fewer than 2 items.
        """
        while True:
            try:
                j_dist, pair = heapq.heappop(self._jdist_table)
                pair = tuple(pair)
            except IndexError:
                return None, None
            if (pair[0] not in self._item_to_set_mask or
                    pair[1] not in self._item_to_set_mask):
                # Items were removed.
                continue
            return pair, j_dist

    def get_all_items(self):
        """Returns all items."""
        return np.array(self._item_to_set_mask.keys())

    def _init_j_dist_table(self, item_to_set_mask):
        dist_table = []
        for item_a, item_b in itertools.combinations(
                item_to_set_mask.keys(), 2):
            j_dist = JDistTable._jaccard_distance(
                item_to_set_mask[item_a],
                item_to_set_mask[item_b])
            dist_table.append((j_dist, frozenset({item_a, item_b})))
        heapq.heapify(dist_table)
        return dist_table

    @staticmethod
    def _jaccard_distance(mask_a, mask_b):
        """Calculates Jaccard distance.

        Args:
            mask_a: Array of bools, a mask that selects the first set.
            mask_a: Array of bools, a mask that selects the second set.

        Returns:
            Float from 0-1, with 0=identical sets, 1=disjoint sets.
        """
        union_len = len(mask_a[mask_a | mask_b])
        intersect_len = len(mask_a[mask_a & mask_b])
        if union_len == 0:
            return 1
        return float(union_len - intersect_len) / union_len
