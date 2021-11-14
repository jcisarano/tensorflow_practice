"""
PCA method that uses the kernel trick, which allows effecient processing of high-dimensional feature sets
"""
from sklearn.datasets import make_swiss_roll
from sklearn.decomposition import KernelPCA
import matplotlib.pyplot as plt

def get_data(n_samples=1000, noise=0.2):
    X, t = make_swiss_roll(n_samples=n_samples, noise=noise, random_state=42)
    return X, t


def create_rbf_kernel_pca(X):
    pca = KernelPCA(n_components=2, kernel="rbf", gamma=0.04)
    X_reduced = pca.fit_transform(X)


def plot_kernel_variations(X, t):
    lin_pca = KernelPCA(n_components=2, kernel="linear", fit_inverse_transform=True)
    rbf_pca = KernelPCA(n_components=2, kernel="rbf", gamma=0.0433, fit_inverse_transform=True)
    sig_pca = KernelPCA(n_components=2, kernel="sigmoid", gamma=0.001, coef0=1, fit_inverse_transform=True)

    y = t > 6.9

    plt.figure(figsize=(11, 4))
    for subplot, pca, title in ((131, lin_pca, "Linear kernel"),
                                (132, rbf_pca, "RBF kernel, $\gamma=0.0433$"),
                                (133, sig_pca, "Sigmoid kernel, $\gamma=10^{-3}, r=1$")):
        X_reduced = pca.fit_transform(X)
        if subplot == 132:
            X_reduced_rbf = X_reduced

        plt.subplot(subplot)
        plt.title(title, fontsize=14)
        plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=t, cmap=plt.cm.hot)
        plt.xlabel("$z_1$", fontsize=18)
        if subplot == 131:
            plt.ylabel("$z_2$", fontsize=18, rotation=0)
        plt.grid(True)

    plt.show()



def run():
    X, t = get_data()

    # rbf_pca = create_rbf_kernel_pca(X)
    plot_kernel_variations(X, t)


