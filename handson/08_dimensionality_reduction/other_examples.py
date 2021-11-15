from sklearn.datasets import make_swiss_roll
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import MDS, Isomap, TSNE


def get_data(n_samples=1000, noise=0.2):
    X, t = make_swiss_roll(n_samples=n_samples, noise=noise, random_state=41)
    return X, t

def run():
    X, t = get_data()

    mds = MDS(n_components=2, random_state=42)
    X_reduced_mds = mds.fit_transform(X)

    isomap = Isomap(n_components=2)
    X_reduced_isomap = isomap.fit_transform(X)

    tsne = TSNE(n_components=2, random_state=42)
    X_reduced_tsne = tsne.fit_transform(X)

    # lda = LinearDiscriminantAnalysis(n_components=2)
