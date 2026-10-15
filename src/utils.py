"""Shared helpers: output paths, logging, seeding, JSON I/O, label encoding."""
import json
import logging
import os
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

import config

SUBDIRS = ["data", "eda", "figures", "tables", "metrics", "predictions",
           "models", "logs", "error_analysis", "stats"]


def out_dir(sub: str) -> Path:
    p = Path(config.OUTPUT_DIR) / sub
    p.mkdir(parents=True, exist_ok=True)
    return p


def out_path(sub: str, name: str) -> Path:
    return out_dir(sub) / name


def get_logger(name: str = "cyberbullying") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    fh = logging.FileHandler(out_path("logs", "run.log"), encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def set_seed(seed: int = None):
    seed = config.SEED if seed is None else seed
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


class _NpEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return None if np.isnan(o) else float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        return super().default(o)


def save_json(obj, path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, cls=_NpEncoder)


def load_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@contextmanager
def timer():
    """Usage: with timer() as t: ...; t() -> elapsed seconds."""
    start = time.perf_counter()
    end = [None]
    yield lambda: (end[0] or time.perf_counter()) - start
    end[0] = time.perf_counter()


# --------------------------------------------------------------------------
# Labels / tasks
# --------------------------------------------------------------------------
def task_labels(task: str):
    return {"binary": config.BINARY_LABELS,
            "harmonised4": config.H4_LABELS,
            "native6": config.TWITTER_NATIVE_LABELS}[task]


def task_label_column(task: str):
    return {"binary": "label_binary", "harmonised4": "label_h4",
            "native6": "label_native"}[task]


def encode_labels(df: pd.DataFrame, task: str) -> np.ndarray:
    labels = task_labels(task)
    mapping = {lab: i for i, lab in enumerate(labels)}
    y = df[task_label_column(task)].map(mapping)
    if y.isna().any():
        bad = df.loc[y.isna(), task_label_column(task)].unique()[:5]
        raise ValueError(f"Unknown labels for task {task}: {bad}")
    return y.to_numpy(dtype=np.int64)


def prepared_path(dataset: str) -> Path:
    return out_path("data", f"{dataset}_prepared.pkl")


def load_prepared(dataset: str) -> pd.DataFrame:
    p = prepared_path(dataset)
    if not p.exists():
        raise FileNotFoundError(f"{p} not found - run the 'prepare' stage first.")
    return pd.read_pickle(p)


def available_prepared():
    return [d for d in config.ACTIVE_DATASETS if prepared_path(d).exists()]


def split_frames(df: pd.DataFrame):
    return (df[df["split"] == "train"], df[df["split"] == "val"],
            df[df["split"] == "test"])


def run_id(train_ds, task, model, variant, eval_ds):
    return f"{train_ds}__{task}__{model}__{variant}__on_{eval_ds}"
