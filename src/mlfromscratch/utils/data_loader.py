from sklearn.datasets import make_regression
from sklearn.datasets import make_classification, make_moons, make_circles
from sklearn.datasets import make_blobs

from sklearn.datasets import load_diabetes
from sklearn.datasets import load_iris, load_wine, load_breast_cancer


def load_regression_data(n_samples=100, n_features=1, noise=10, random_state=42):
    """
    Load synthetic regression dataset.

    Args:
        n_samples (int): Number of samples to generate
        n_features (int): Number of features
        noise (float): Standard deviation of the gaussian noise
        random_state (int): Random state for reproducibility

    Returns:
        tuple: (X, y) where X is features and y is targets
    """
    X, y = make_regression(n_samples=n_samples, n_features=n_features,
                          noise=noise, random_state=random_state)
    return X, y


def load_classification_data(n_samples=100, n_features=2, n_classes=2,
                           n_redundant=0, random_state=42):
    """
    Load synthetic classification dataset.

    Args:
        n_samples (int): Number of samples to generate
        n_features (int): Number of features
        n_classes (int): Number of classes
        n_redundant (int): Number of redundant features
        random_state (int): Random state for reproducibility

    Returns:
        tuple: (X, y) where X is features and y is class labels
    """
    X, y = make_classification(n_samples=n_samples, n_features=n_features,
                              n_classes=n_classes, n_redundant=n_redundant,
                              random_state=random_state)
    return X, y


def load_moons_data(n_samples=100, noise=0.1, random_state=42):
    """
    Load two interleaving half circles dataset.

    Args:
        n_samples (int): Number of samples to generate
        noise (float): Standard deviation of gaussian noise
        random_state (int): Random state for reproducibility

    Returns:
        tuple: (X, y) where X is features and y is binary labels
    """
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    return X, y


def load_circles_data(n_samples=100, noise=0.1, factor=0.8, random_state=42):
    """
    Load two concentric circles' dataset.

    Args:
        n_samples (int): Number of samples to generate
        noise (float): Standard deviation of gaussian noise
        factor (float): Scale factor between inner and outer circle
        random_state (int): Random state for reproducibility

    Returns:
        tuple: (X, y) where X is features and y is binary labels
    """
    X, y = make_circles(n_samples=n_samples, noise=noise, factor=factor,
                       random_state=random_state)
    return X, y


def load_blobs_data(n_samples=100, centers=3, n_features=2, random_state=42):
    """
    Load isotropic Gaussian blobs dataset.

    Args:
        n_samples (int): Number of samples to generate
        centers (int): Number of centers to generate
        n_features (int): Number of features
        random_state (int): Random state for reproducibility

    Returns:
        tuple: (X, y) where X is features and y is cluster labels
    """
    X, y = make_blobs(n_samples=n_samples, centers=centers, n_features=n_features,
                     random_state=random_state)
    return X, y


def load_diabetes_data():
    """
    Load diabetes regression dataset.

    Returns:
        tuple: (X, y) where X is features and y is diabetes progression targets
    """
    diabetes = load_diabetes()
    return diabetes.data, diabetes.target


def load_iris_data():
    """
    Load iris classification dataset.

    Returns:
        tuple: (X, y) where X is features and y is species labels
    """
    iris = load_iris()
    return iris.data, iris.target


def load_wine_data():
    """
    Load wine classification dataset.

    Returns:
        tuple: (X, y) where X is features and y is wine class labels
    """
    wine = load_wine()
    return wine.data, wine.target


def load_breast_cancer_data():
    """
    Load breast cancer classification dataset.

    Returns:
        tuple: (X, y) where X is features and y is binary labels (malignant/benign)
    """
    cancer = load_breast_cancer()
    return cancer.data, cancer.target


def get_available_datasets():
    """
    Get list of available dataset loading functions.

    Returns:
        dict: Dictionary mapping dataset names to their loading functions
    """
    return {
        'regression': load_regression_data,
        'classification': load_classification_data,
        'moons': load_moons_data,
        'circles': load_circles_data,
        'blobs': load_blobs_data,
        'diabetes': load_diabetes_data,
        'iris': load_iris_data,
        'wine': load_wine_data,
        'breast_cancer': load_breast_cancer_data
    }
