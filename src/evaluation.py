"""Standardised evaluation (proposal: accuracy, precision, recall, F1, AUC-ROC,
confusion matrix and statistical significance).

Every model calls `record_result`, which writes
  outputs/metrics/<run_id>.json      - all metrics + metadata
  outputs/predictions/<run_id>.npz   - y_true, y_pred, probabilities
  outputs/figures/...                - confusion matrix and ROC curve
so that later stages (statistics, report) can be run independently, even if
the transformers were trained on a different machine (e.g. Google Colab).
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (accuracy_score, auc, confusion_matrix,
                             precision_recall_fscore_support, roc_auc_score,
                             roc_curve)
from sklearn.preprocessing import label_binarize

import config
from src import utils

log = utils.get_logger()


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def _normalise_proba(proba):
    proba = np.asarray(proba, dtype=np.float64)
    s = proba.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    return proba / s


def auc_roc(y_true, proba, n_classes):
    try:
        if n_classes == 2:
            return float(roc_auc_score(y_true, proba[:, 1]))
        present = np.unique(y_true)
        if len(present) < 2:
            return float("nan")
        return float(roc_auc_score(y_true, _normalise_proba(proba), multi_class="ovr",
                                   average="macro", labels=list(range(n_classes))))
    except ValueError:
        return float("nan")


def macro_f1_fast(y_true, y_pred, n_classes):
    """Macro F1 from a confusion matrix (vectorised; used in bootstraps)."""
    cm = np.bincount(n_classes * y_true + y_pred, minlength=n_classes ** 2).reshape(n_classes, n_classes)
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    denom = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    return f1.mean()


def bootstrap_ci(y_true, y_pred, n_classes, iters=None, seed=None):
    iters = iters or config.BOOTSTRAP_ITERATIONS
    rng = np.random.default_rng(config.SEED if seed is None else seed)
    n = len(y_true)
    scores = np.empty(iters)
    for i in range(iters):
        idx = rng.integers(0, n, n)
        scores[i] = macro_f1_fast(y_true[idx], y_pred[idx], n_classes)
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def compute_metrics(y_true, y_pred, proba, labels):
    n = len(labels)
    lab_idx = list(range(n))
    p_m, r_m, f_m, _ = precision_recall_fscore_support(y_true, y_pred, labels=lab_idx,
                                                       average="macro", zero_division=0)
    p_w, r_w, f_w, _ = precision_recall_fscore_support(y_true, y_pred, labels=lab_idx,
                                                       average="weighted", zero_division=0)
    p_c, r_c, f_c, s_c = precision_recall_fscore_support(y_true, y_pred, labels=lab_idx,
                                                         average=None, zero_division=0)
    ci_lo, ci_hi = bootstrap_ci(y_true, y_pred, n)
    m = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": p_m, "recall_macro": r_m, "f1_macro": f_m,
        "f1_macro_ci_low": ci_lo, "f1_macro_ci_high": ci_hi,
        "precision_weighted": p_w, "recall_weighted": r_w, "f1_weighted": f_w,
        "auc_roc": auc_roc(y_true, proba, n),
        "per_class": {lab: {"precision": float(p_c[i]), "recall": float(r_c[i]),
                            "f1": float(f_c[i]), "support": int(s_c[i])}
                      for i, lab in enumerate(labels)},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=lab_idx).tolist(),
    }
    if n == 2:  # positive class = cyberbullying
        m.update({"precision_pos": float(p_c[1]), "recall_pos": float(r_c[1]), "f1_pos": float(f_c[1])})
    return {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in m.items()}


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------
def plot_confusion(cm, labels, title, path):
    cm = np.asarray(cm)
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(1.3 * len(labels) + 3, 1.1 * len(labels) + 2.5))
    annot = np.array([[f"{c:,}\n{p:.1%}" for c, p in zip(rc, rp)] for rc, rp in zip(cm, norm)])
    sns.heatmap(norm, annot=annot, fmt="", cmap="Blues", vmin=0, vmax=1, ax=ax,
                xticklabels=labels, yticklabels=labels, cbar=False, annot_kws={"fontsize": 8})
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title, fontsize=10)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_roc(y_true, proba, labels, title, path):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    if len(labels) == 2:
        fpr, tpr, _ = roc_curve(y_true, proba[:, 1])
        ax.plot(fpr, tpr, label=f"{labels[1]} (AUC={auc(fpr, tpr):.3f})")
    else:
        yb = label_binarize(y_true, classes=list(range(len(labels))))
        for i, lab in enumerate(labels):
            if yb[:, i].sum() == 0:
                continue
            fpr, tpr, _ = roc_curve(yb[:, i], proba[:, i])
            ax.plot(fpr, tpr, label=f"{lab} (AUC={auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# Recording
# --------------------------------------------------------------------------
def record_result(*, train_ds, eval_ds, task, model, variant, is_main, y_true, proba,
                  labels, y_pred=None, extra=None, make_plots=True):
    """y_pred defaults to argmax(proba); pass it explicitly when a model's class
    decision differs from its (calibrated) probabilities, e.g. the SVM."""
    y_true = np.asarray(y_true, dtype=np.int64)
    proba = _normalise_proba(proba)
    y_pred = proba.argmax(axis=1) if y_pred is None else np.asarray(y_pred, dtype=np.int64)
    rid = utils.run_id(train_ds, task, model, variant, eval_ds)
    metrics = compute_metrics(y_true, y_pred, proba, labels)
    record = {
        "run_id": rid, "train_dataset": train_ds, "eval_dataset": eval_ds,
        "cross_dataset": train_ds != eval_ds, "task": task, "model": model,
        "model_family": config.MODEL_FAMILIES.get(model, "other"), "variant": variant,
        "is_main": bool(is_main), "labels": list(labels), "n_test": int(len(y_true)),
        "quick_mode": config.QUICK_MODE, **metrics, **(extra or {}),
    }
    utils.save_json(record, utils.out_path("metrics", f"{rid}.json"))
    np.savez_compressed(utils.out_path("predictions", f"{rid}.npz"),
                        y_true=y_true, y_pred=y_pred, proba=proba.astype(np.float32))
    if make_plots and (is_main or train_ds != eval_ds):
        name = config.MODEL_DISPLAY.get(model, model)
        title = f"{name} | trained {train_ds} -> tested {eval_ds} | {task}"
        plot_confusion(metrics["confusion_matrix"], labels, title,
                       utils.out_path("figures", f"cm__{rid}.png"))
        if train_ds == eval_ds:
            plot_roc(y_true, proba, labels, title, utils.out_path("figures", f"roc__{rid}.png"))
    log.info(f"   -> {rid}: acc={metrics['accuracy']:.4f} F1(macro)={metrics['f1_macro']:.4f} "
             f"AUC={metrics['auc_roc']:.4f}")
    return record


def cross_dataset_targets(train_ds):
    """Other prepared datasets on which binary models are also tested."""
    return [d for d in utils.available_prepared() if d != train_ds]
