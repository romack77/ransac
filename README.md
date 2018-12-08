# RANSAC

RANSAC fits models to noisy data.

It is most useful when the data contains outliers that should be ignored completely
during model fitting
([Wikipedia RANSAC page](https://en.wikipedia.org/wiki/Random_sample_consensus)).

## Basic Usage

```
import ransac
my_ransac = ransac.Ransac(
    model, num_sample_points, min_inliers, inlier_threshold, stop_iterations)
results = my_ransac.run(list_of_data_points)
print(results.fit, results.inliers, results.outliers)
```

While ```ransac.Ransac``` always fits a single model, ```ransac.XRansac``` or ```ransac.JLinkage```
can be used to detect and fit multiple underlying models. The number of models does not need to
be specified in advance. XRansac is faster but uses additional parameters.

See [this IPython notebook](https://github.com/romack77/ransac/blob/master/ransac/notebooks/RansacExample.ipynb)
for more complete examples.

To run the IPython notebooks locally:
```
make ipython
```

#### Parameter tuning

Note that the various RANSAC threshold parameters can have a large effect on the results and
will likely need to be tuned for any particular use case. In particular, inlier_threshold
determines what is considered an inlier, while stop_iterations presents a trade
off between accuracy and speed. If the number of outliers can be estimated,
`ransac.calculate_ransac_iterations` can help choose a good stop_iterations value.

#### Custom models
To use a custom model, subclass `ransac.Model` and implement the `fit` and `predict` methods.
See [ransac/models/base.py](https://github.com/romack77/ransac/blob/master/ransac/models/base.py)
for more details.


## Installation

Requires Python 3 and Make.

To install:
```
make
```
To run tests:
```
make test
```

## License

This project is licensed under the MIT License - see the [LICENSE.txt](LICENSE.txt) file for details.

## References

This paper is the basis of the XRansac variant (no affiliation):

Zhang, W., Kosecká, J.: Nonparametric estimation of multiple structures with outliers. In: Vidal, R., Heyden, A., Ma, Y. (eds.) WDV 2006. LNCS, vol. 4358, pp. 60–74. Springer, Heidelberg (2006)
[PDF](https://link.springer.com/chapter/10.1007/978-3-540-70932-9_5)

This paper is the basis of the J-linkage variant (no affiliation):

Toldo, R., & Fusiello, A. (2008, October). Robust multiple structures estimation with j-linkage. In European conference on computer vision (pp. 537-547). Springer, Berlin, Heidelberg.
[PDF](https://link.springer.com/chapter/10.1007/978-3-540-88682-2_41)