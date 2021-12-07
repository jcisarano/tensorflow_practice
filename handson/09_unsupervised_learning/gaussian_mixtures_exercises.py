import numpy as np
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt

from visualization_helpers import plot_gaussian_mixture


def get_blob_data():
    x1, y1 = make_blobs(n_samples=1000, centers=((4, -4), (0, 0)), random_state=42)
    x1 = x1.dot(np.array([[0.374, 0.95], [0.732, 0.598]]))
    x2, y2 = make_blobs(n_samples=250, centers=1, random_state=42)
    x2 = x2 + [6, -8]
    X = np.r_[x1, x2]
    y = np.r_[y1, y2]

    return X, y


def examine_gm(X, y):
    gm = GaussianMixture(n_components=3, n_init=10, random_state=42)
    gm.fit(X)

    print("GM weights:", gm.weights_)
    print("GM means:", gm.means_)
    print("GM covariances:", gm.covariances_)
    print("GM converged:", gm.converged_)
    print("GM num iterations:", gm.n_iter_)
    print("GM predictions:", gm.predict(X))
    print("GM pred probs:", gm.predict_proba(X))

    # this model can create new instances along with their labels:
    X_new, y_new = gm.sample(6)
    print("Generated instances:", X_new)
    print("Generated labels:", y_new)

    # log of the Probability Density Function
    print("PDF:", gm.score_samples(X))

    # check that PDF integrates to 1 over the entire space:
    # first, create a grid of tiny squares
    resolution = 100
    grid = np.arange(-10, 10, 1 / resolution)
    xx, yy = np.meshgrid(grid, grid)
    X_full = np.vstack([xx.ravel(), yy.ravel()]).T

    pdf = np.exp(gm.score_samples(X_full))
    pdf_probas = pdf * (1 / resolution) ** 2  # multiply pdf by area of its square
    print("Close to 1:", pdf_probas.sum())

    plt.figure(figsize=(8, 4))
    plot_gaussian_mixture(gm, X)
    plt.show()


def examine_var_gm(X, y):
    """
    examine different GM covariance type settings
    full: any ellipsoid cluster shape allowed
    tied: all clusters must have the same shape
    spherical: all clusters must be spherical
    diag: clusters must have axes parallel to axes (which makes the covariance diagonal)
    :param X:
    :param y:
    :return:
    """
    gm_full = GaussianMixture(n_components=3, n_init=10, covariance_type="full", random_state=42)
    gm_tied = GaussianMixture(n_components=3, n_init=10, covariance_type="tied", random_state=42)
    gm_spherical = GaussianMixture(n_components=3, n_init=10, covariance_type="spherical", random_state=42)
    gm_diag = GaussianMixture(n_components=3, n_init=10, covariance_type="diag", random_state=42)
    gm_full.fit(X)
    gm_tied.fit(X)
    gm_spherical.fit(X)
    gm_diag.fit(X)

    compare_gaussian_mixtures(gm_tied, gm_spherical, X)
    plt.show()

    compare_gaussian_mixtures(gm_full, gm_diag, X)
    plt.tight_layout()
    plt.show()


def compare_gaussian_mixtures(gm1, gm2, X):
    plt.figure(figsize=(9, 4))
    plt.subplot(121)
    plot_gaussian_mixture(gm1, X)
    plt.title("covariance_type={}".format(gm1.covariance_type), fontsize=14)

    plt.subplot(122)
    plot_gaussian_mixture(gm2, X)
    plt.title("covariance_type={}".format(gm2.covariance_type), fontsize=14)


def anomaly_detection(X, y):
    gm = GaussianMixture(n_components=3, n_init=10, random_state=42)
    gm.fit(X)

    densities = gm.score_samples(X)
    # calculate value for 4th percentile
    density_threshold = np.percentile(densities, 4)
    # anomalies are any X where score_sample is less than the 4th percentile threshold
    anomalies = X[densities < density_threshold]

    plt.figure(figsize=(8, 4))
    plot_gaussian_mixture(gm, X)
    plt.scatter(anomalies[:, 0], anomalies[:, 1], color="r", marker="*")
    plt.ylim(top=5.1)
    plt.show()


def select_num_clusters(X, y):
    gm = GaussianMixture(n_components=3, n_init=10, random_state=42)
    gm.fit(X)

    print("gm BIC", gm.bic(X))
    print("gm AIC", gm.aic(X))

    # manual calculat BIC and AIC
    n_clusters = 3
    n_dims = 2
    n_params_for_weights = n_clusters - 1
    n_params_for_means = n_clusters * n_dims
    n_params_for_covariance = n_clusters * n_dims * (n_dims + 1) // 2
    n_params = n_params_for_weights + n_params_for_means + n_params_for_covariance
    max_log_likelihood = gm.score(X) * len(X)  # Log(L^)
    bic = np.log(len(X)) * n_params - 2 * max_log_likelihood
    aic = 2 * n_params - 2 * max_log_likelihood

    print("Manual BIC: {}, AIC: {}".format(bic, aic))
    print("Num params:", n_params)


def train_for_k_vals(X, y):
    gms_per_k = [GaussianMixture(n_components=k, n_init=10, random_state=42).fit(X)
                 for k in range(1, 11)]
    bics = [model.bic(X) for model in gms_per_k]
    aics = [model.aic(X) for model in gms_per_k]

    plt.figure(figsize=(8, 3))
    plt.plot(range(1, 11), bics, "bo-", label="BIC")
    plt.plot(range(1, 11), aics, "go--", label="AIC")
    plt.xlabel("$k$", fontsize=14)
    plt.ylabel("$Information Criterion", fontsize=14)
    plt.axis([1, 9.5, np.min(aics)-5, np.max(aics)+50])
    plt.annotate("Minimum",
                 xy=(3, bics[2]),
                 xytext=(0.35, 0.6),
                 textcoords="figure fraction",
                 fontsize=14,
                 arrowprops=dict(facecolor="black", shrink=0.1)
                 )
    plt.legend()
    plt.show()


def find_best_covariance_type_and_k(X, y):
    min_bic = np.infty

    for k in range(1, 11):
        for covariance_type in ("full", "tied", "spherical", "diag"):
            bic = GaussianMixture(n_components=k,
                                  n_init=10,
                                  covariance_type=covariance_type,
                                  random_state=42).fit(X).bic(X)
            if bic < min_bic:
                min_bic = bic
                best_k = k
                best_covariance_type = covariance_type
    print("Best k:", best_k)
    print("Best covariance type:", best_covariance_type)


def run():
    X, y = get_blob_data()
    # examine_gm(X, y)
    # examine_var_gm(X, y)
    # anomaly_detection(X, y)
    # select_num_clusters(X, y)
    # train_for_k_vals(X, y)
    find_best_covariance_type_and_k(X, y)


