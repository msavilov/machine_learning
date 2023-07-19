from sklearn.datasets import fetch_openml


def load_df():
    mnist = fetch_openml('mnist_784', as_frame=False)
    return mnist
