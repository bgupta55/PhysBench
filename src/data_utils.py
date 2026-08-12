"""
Data utilities for PhysBench experiments.
Supports both file-based loading (CIFAR-10-images mirror) and torchvision-based
automatic download as a fallback, so experiments run without manual data prep.
"""
import os
import glob
import random
import numpy as np
import torch

CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]

# Standard CIFAR-10 channel normalization stats
MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32).reshape(1, 3, 1, 1)
STD  = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32).reshape(1, 3, 1, 1)


def _try_load_from_files(root, split, per_class, seed):
    """Try to load from YoongiKim/CIFAR-10-images style file tree."""
    rng = random.Random(seed)
    X, y = [], []
    from PIL import Image
    for ci, cname in enumerate(CLASSES):
        files = sorted(glob.glob(os.path.join(root, split, cname, "*.jpg")))
        if not files:
            files = sorted(glob.glob(os.path.join(root, split, cname, "*.png")))
        if not files:
            return None, None  # signal failure
        rng.shuffle(files)
        chosen = files[:per_class]
        for f in chosen:
            img = Image.open(f).convert("RGB")
            arr = np.asarray(img, dtype=np.float32) / 255.0
            X.append(arr.transpose(2, 0, 1))
            y.append(ci)
    X = np.stack(X).astype(np.float32)
    y = np.array(y, dtype=np.int64)
    X = (X - MEAN) / STD
    return torch.from_numpy(X), torch.from_numpy(y)


def _load_from_torchvision(split, per_class, seed):
    """Load CIFAR-10 via torchvision (auto-downloads to ~/.cache/cifar10)."""
    import torchvision
    import torchvision.transforms as T

    cache_dir = os.path.expanduser("~/.cache/cifar10")
    os.makedirs(cache_dir, exist_ok=True)
    is_train = (split == "train")
    ds = torchvision.datasets.CIFAR10(root=cache_dir, train=is_train, download=True)

    # Organize by class
    class_images = {c: [] for c in range(10)}
    for img, label in ds:
        arr = np.asarray(img, dtype=np.float32) / 255.0
        class_images[label].append(arr.transpose(2, 0, 1))

    rng = random.Random(seed)
    X, y = [], []
    for ci in range(10):
        imgs = class_images[ci]
        rng.shuffle(imgs)
        chosen = imgs[:per_class]
        for arr in chosen:
            X.append(arr)
            y.append(ci)

    X = np.stack(X).astype(np.float32)
    y = np.array(y, dtype=np.int64)
    X = (X - MEAN) / STD
    return torch.from_numpy(X), torch.from_numpy(y)


# Default file-tree root (original pipeline location; overridden if missing)
_FILE_ROOT = os.environ.get("CIFAR10_ROOT", "/home/claude/data/CIFAR-10-images-master")


def get_data(train_per_class=300, test_per_class=100, seed=0):
    """Return (Xtr, ytr, Xte, yte) tensors, shape [N,3,32,32] and [N]."""
    Xtr, ytr = _try_load_from_files(_FILE_ROOT, "train", train_per_class, seed)
    Xte, yte = _try_load_from_files(_FILE_ROOT, "test",  test_per_class,  seed + 1000)

    if Xtr is None:
        # Fall back to torchvision download
        print("[data_utils] File tree not found; downloading CIFAR-10 via torchvision…")
        Xtr, ytr = _load_from_torchvision("train", train_per_class, seed)
        Xte, yte = _load_from_torchvision("test",  test_per_class,  seed + 1000)
    return Xtr, ytr, Xte, yte
