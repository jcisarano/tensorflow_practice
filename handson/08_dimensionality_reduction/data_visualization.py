import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_swiss_roll
from sklearn.decomposition import PCA

import data_utils as du


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


def plot_3d_dataset_close_to_2d_subspace(pca, X, X3D_inv, m):
    axes = [-1.8, 1.8, -1.3, 1.3, -1.0, 1.0]

    # express plane as function of x and y
    x1s = np.linspace(axes[0], axes[1], 10)
    x2s = np.linspace(axes[2], axes[3], 10)
    x1, x2 = np.meshgrid(x1s, x2s)

    C = pca.components_
    R = C.T.dot(C)
    z = (R[0, 2] * x1 + R[1, 2] * x2) / (1 - R[2, 2])

    fig = plt.figure(figsize=(6, 3.8))
    ax = fig.add_subplot(111, projection="3d")

    X3D_above = X[X[:, 2] > X3D_inv[:, 2]]
    X3D_below = X[X[:, 2] <= X3D_inv[:, 2]]

    ax.plot(X3D_below[:, 0], X3D_below[:, 1], X3D_below[:, 2], "bo", alpha=0.5)

    ax.plot_surface(x1, x2, z, alpha=0.2, color="k")
    np.linalg.norm(C, axis=0)
    ax.add_artist(
        Arrow3D([0, C[0, 0]], [0, C[0, 1]], [0, C[0, 2]], mutation_scale=15, lw=1, arrowstyle="-|>", color="k"))
    ax.add_artist(
        Arrow3D([0, C[1, 0]], [0, C[1, 1]], [0, C[1, 2]], mutation_scale=15, lw=1, arrowstyle="-|>", color="k"))
    ax.plot([0], [0], [0], "k.")

    for i in range(m):
        if X[i, 2] > X3D_inv[i, 2]:
            ax.plot([X[i][0], X3D_inv[i][0]], [X[i][1], X3D_inv[i][1]], [X[i][2], X3D_inv[i][2]], "k-")
        else:
            ax.plot([X[i][0], X3D_inv[i][0]], [X[i][1], X3D_inv[i][1]], [X[i][2], X3D_inv[i][2]], "-", color="#505050")

    ax.plot(X3D_inv[:, 0], X3D_inv[:, 1], X3D_inv[:, 2], "k+")
    ax.plot(X3D_inv[:, 0], X3D_inv[:, 1], X3D_inv[:, 2], "k.")
    ax.plot(X3D_above[:, 0], X3D_above[:, 1], X3D_above[:, 2], "bo")
    ax.set_xlabel("$x_1$", fontsize=18, labelpad=10)
    ax.set_ylabel("$y_2$", fontsize=18, labelpad=10)
    ax.set_zlabel("$z_3$", fontsize=18, labelpad=10)
    ax.set_xlim(axes[0:2])
    ax.set_ylim(axes[2:4])
    ax.set_zlim(axes[4:6])

    plt.show()


def plot_2d_dataset_projection(X2D):
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect="equal")

    ax.plot(X2D[:, 0], X2D[:, 1], "k+")
    ax.plot(X2D[:, 0], X2D[:, 1], "k.")
    ax.arrow(0, 0, 0, 1, head_width=0.05, length_includes_head=True, head_length=0.1, fc="k", ec="k")
    ax.arrow(0, 0, 1, 0, head_width=0.05, length_includes_head=True, head_length=0.1, fc="k", ec="k")
    ax.set_xlabel("$x_1$", fontsize=18)
    ax.set_ylabel("$z_2$", fontsize=18, rotation=0)
    ax.axis([-1.5, 1.3, -1.2, 1.2])
    ax.grid(True)

    plt.show()


def plot_swiss_roll(X, t):
    axes = [-11.5, 14, -2, 23, -12, 15]
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=t, cmap=plt.cm.hot)
    ax.view_init(10, -70)
    ax.set_xlabel("$x_1$", fontsize=18)
    ax.set_ylabel("$x_2$", fontsize=18)
    ax.set_zlabel("$x_3$", fontsize=18)
    ax.set_xlim(axes[0:2])
    ax.set_ylim(axes[2:4])
    ax.set_zlim(axes[4:6])

    plt.show()


def plot_competing_swiss_roll_squashes(X, t):
    axes = [-11.5, 14, -2, 23, -12, 15]
    plt.figure(figsize=(11, 4))

    plt.subplot(121)
    plt.scatter(X[:, 0], X[:, 1], c=t, cmap=plt.cm.hot)
    plt.axis(axes[:4])
    plt.xlabel("$x_1$", fontsize=18)
    plt.ylabel("$x_2$", fontsize=18)
    plt.grid(True)

    plt.subplot(122)
    plt.scatter(t, X[:, 1], c=t, cmap=plt.cm.hot)
    plt.axis([4, 15, axes[2], axes[3]])
    plt.xlabel("$z_1$", fontsize=18)
    plt.grid(True)

    plt.show()


def plot_complex_decision_boundaries(X, t):
    axes = [-11.5, 14, -2, 23, -12, 15]
    x2s = np.linspace(axes[2], axes[3], 10)
    x3s = np.linspace(axes[4], axes[5], 10)
    x2, x3 = np.meshgrid(x2s, x3s)

    # plot boundary through 3d roll
    fig = plt.figure(figsize=(6, 5))
    ax = plt.subplot(111, projection='3d')

    positive_class = X[:, 0] > 5
    X_pos = X[positive_class]
    X_neg = X[~positive_class]
    ax.view_init(10, -70)
    ax.plot(X_neg[:, 0], X_neg[:, 1], X_neg[:, 2], "y^")
    ax.plot_wireframe(5, x2, x3, alpha=0.5)
    ax.plot(X_pos[:, 0], X_pos[:, 1], X_pos[:, 2], "gs")
    ax.set_xlabel("$x_1$", fontsize=18)
    ax.set_ylabel("$x_2$", fontsize=18)
    ax.set_zlabel("$x_3$", fontsize=18)
    ax.set_xlim(axes[0:2])
    ax.set_ylim(axes[2:4])
    ax.set_zlim(axes[4:6])

    # plot manifold boundary 1
    fig = plt.figure(figsize=(5, 4))
    ax = plt.subplot(111)

    plt.plot(t[positive_class], X[positive_class, 1], "gs")
    plt.plot(t[~positive_class], X[~positive_class, 1], "y^")
    plt.axis([4, 15, axes[2], axes[3]])
    plt.xlabel("$z_1$", fontsize=18)
    plt.ylabel("$z_2$", fontsize=18, rotation=0)
    plt.grid(True)

    fig = plt.figure(figsize=(6, 5))
    ax = plt.subplot(111, projection="3d")

    positive_class = 2*(t[:]-4) > X[:, 1]
    X_pos = X[positive_class]
    X_neg = X[~positive_class]
    ax.view_init(10, -70)
    ax.plot(X_neg[:, 0], X_neg[:, 1], X_neg[:, 2], "y^")
    ax.plot(X_pos[:, 0], X_pos[:, 1], X_pos[:, 2], "gs")
    ax.set_xlabel("$x_1$", fontsize=18)
    ax.set_ylabel("$x_2$", fontsize=18)
    ax.set_zlabel("$x_3$", fontsize=18)
    ax.set_xlim(axes[0:2])
    ax.set_ylim(axes[2:4])
    ax.set_zlim(axes[4:6])

    fig = plt.figure(figsize=(5, 4))
    ax = plt.subplot(111)

    plt.plot(t[positive_class], X[positive_class, 1], "gs")
    plt.plot(t[~positive_class], X[~positive_class, 1], "y^")
    plt.plot([4, 15], [0, 22], "b-", linewidth=2)
    plt.axis([4, 15, axes[2], axes[3]])
    plt.xlabel("$z_1$", fontsize=18)
    plt.ylabel("$z_2$", fontsize=18, rotation=0)
    plt.grid(True)


    plt.show()




def run():
    # X = du.get_3d_dataset()
    # m, n = X.shape

    # pca = PCA(n_components=2)
    # X2D = pca.fit_transform(X)
    # X3D_inv = pca.inverse_transform(X2D)

    # plot_3d_dataset_close_to_2d_subspace(pca, X, X3D_inv, m)
    # plot_2d_dataset_projection(X2D)

    X, t = make_swiss_roll(n_samples=1000, noise=0.2, random_state=42)
    # plot_swiss_roll(X, t)
    # plot_competing_swiss_roll_squashes(X, t)
    plot_complex_decision_boundaries(X, t)

