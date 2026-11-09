import re
from pathlib import Path

import matplotlib.image as mpimg
import numpy as np


def make_xor():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 1, 1, 0], dtype=float)
    return X, y


def make_moons(n_samples=200, noise=0.1, seed=0):
    rng = np.random.default_rng(seed)
    n_top = n_samples // 2
    n_bottom = n_samples - n_top
    t_top = rng.uniform(0, np.pi, n_top)
    t_bottom = rng.uniform(0, np.pi, n_bottom)
    top = np.column_stack([np.cos(t_top), np.sin(t_top)])
    bottom = np.column_stack([1 - np.cos(t_bottom), 0.5 - np.sin(t_bottom)])
    X = np.vstack([top, bottom]) + rng.normal(0, noise, (n_samples, 2))
    y = np.concatenate([np.zeros(n_top), np.ones(n_bottom)])
    return X, y


def save_csv(path, X, y):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    X = np.asarray(X, dtype=float)
    header = ",".join([f"x{i}" for i in range(X.shape[1])] + ["y"])
    np.savetxt(path, np.column_stack([X, y]), delimiter=",", header=header, comments="")
    return path


def load_csv(path):
    data = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    return data[:, :-1], data[:, -1]


def load_image_folder(folder, pattern=r"digit_(\d+)_idx_\d+"):
    files = sorted(f for f in Path(folder).glob("*.png") if re.fullmatch(pattern, f.stem))
    if not files:
        raise FileNotFoundError(f"no .png files named like {pattern} in {folder}")

    images = []
    for f in files:
        image = mpimg.imread(f)
        if image.ndim == 3:
            image = image[..., :3].mean(axis=2)
        images.append(image.reshape(-1))
    X = np.stack(images).astype(float)
    y = np.array([int(re.fullmatch(pattern, f.stem).group(1)) for f in files])
    return X, y


def one_hot(y, n_classes):
    return np.eye(n_classes)[np.asarray(y, dtype=int)]


def train_test_split(X, y, test_size=0.2, seed=0):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    order = np.random.default_rng(seed).permutation(len(X))
    n_test = int(round(len(X) * test_size))
    test, train = order[:n_test], order[n_test:]
    return X[train], X[test], y[train], y[test]
