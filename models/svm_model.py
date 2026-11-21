import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


def generate_linear_data(n_samples=100, random_state=42):
    X, y = make_blobs(n_samples=n_samples, centers=2, random_state=random_state, cluster_std=1.0)
    return X, y


def generate_moon_data(n_samples=100, noise=0.2, random_state=42):
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    return X, y


def generate_circular_data(n_samples=100, noise=0.1, random_state=42):
    X, y = make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=random_state)
    return X, y


def train_svm(X, y, kernel='linear', C=1.0, gamma='scale', degree=3):
    clf = svm.SVC(kernel=kernel, C=C, gamma=gamma, degree=degree)
    clf.fit(X, y)
    return clf


def plot_decision_boundary(clf, X, y, filename):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
    plt.title(f'SVM Decision Boundary - Kernel: {clf.kernel}')
    plt.savefig(filename)
    plt.close()
    return filename
