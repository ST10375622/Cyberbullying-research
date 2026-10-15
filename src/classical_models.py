"""Models 1-3: Logistic Regression, Support Vector Machine, Random Forest on
TF-IDF (uni+bi-gram) features.

* Hyper-parameters are selected on the validation split (macro F1).
* Main runs use SMOTE on the training split only (proposal).
* Imbalance ablation (SQ4): each model is also trained with no rebalancing
  and with class weighting.
* Binary models are additionally tested on the other dataset's test split
  (cross-dataset benchmarking) and saved for deployment.
"""
import time

import joblib
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import ParameterGrid
from sklearn.svm import LinearSVC

import config
from src import utils
from src.evaluation import cross_dataset_targets, record_result

log = utils.get_logger()


class CalibratedLinearSVC(BaseEstimator, ClassifierMixin):
    """Linear SVM whose class predictions come from the SVM decision function,
    with sigmoid-calibrated probabilities used only for AUC-ROC.

    Taking argmax of calibrated probabilities would re-learn the class priors and
    cancel the effect of class weighting, which would bias the imbalance ablation."""

    def __init__(self, C=1.0, class_weight=None):
        self.C = C
        self.class_weight = class_weight

    def _svm(self):
        return LinearSVC(C=self.C, class_weight=self.class_weight, random_state=config.SEED)

    def fit(self, X, y):
        self.svm_ = self._svm().fit(X, y)
        self.calibrated_ = CalibratedClassifierCV(self._svm(), cv=3, method="sigmoid").fit(X, y)
        self.classes_ = self.svm_.classes_
        return self

    def predict(self, X):
        return self.svm_.predict(X)

    def predict_proba(self, X):
        return self.calibrated_.predict_proba(X)


def build_estimator(name, params, class_weight=None):
    if name == "logreg":
        return LogisticRegression(C=params["C"], max_iter=3000, class_weight=class_weight,
                                  random_state=config.SEED)
    if name == "svm":
        return CalibratedLinearSVC(C=params["C"], class_weight=class_weight)
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=params["n_estimators"], min_samples_leaf=params["min_samples_leaf"],
            max_features="sqrt", class_weight="balanced_subsample" if class_weight else None,
            n_jobs=-1, random_state=config.SEED)
    raise ValueError(name)


def apply_smote(X, y):
    counts = np.bincount(y)
    majority = counts.max()
    target = {c: int(config.SMOTE_TARGET_RATIO * majority) for c, n in enumerate(counts)
              if 2 <= n < config.SMOTE_TARGET_RATIO * majority}
    if not target:
        return X, y, {}
    k = max(1, min(config.SMOTE_K_NEIGHBORS, min(counts[c] for c in target) - 1))
    sm = SMOTE(sampling_strategy=target, k_neighbors=k, random_state=config.SEED)
    Xr, yr = sm.fit_resample(X, y)
    info = {"before": counts.tolist(), "after": np.bincount(yr).tolist(), "k_neighbors": k}
    return Xr, yr, info


def tune_and_fit(name, X_fit, y_fit, X_val, y_val, class_weight):
    best = None
    for params in ParameterGrid(config.CLASSICAL_GRIDS[name]):
        est = build_estimator(name, params, class_weight)
        t0 = time.perf_counter()
        est.fit(X_fit, y_fit)
        fit_time = time.perf_counter() - t0
        f1 = f1_score(y_val, est.predict(X_val), average="macro")
        log.info(f"     {name} {params} val F1={f1:.4f} ({fit_time:.1f}s)")
        if best is None or f1 > best[1]:
            best = (est, f1, params, fit_time)
    return best


def run_task(dataset, task, df, models):
    labels = utils.task_labels(task)
    tr, va, te = utils.split_frames(df)
    y_tr, y_va, y_te = (utils.encode_labels(x, task) for x in (tr, va, te))

    vec = TfidfVectorizer(**config.TFIDF_PARAMS)
    X_tr = vec.fit_transform(tr["text_proc"])
    X_va, X_te = vec.transform(va["text_proc"]), vec.transform(te["text_proc"])
    log.info(f"[classical] {dataset}/{task}: TF-IDF {X_tr.shape}, classes={np.bincount(y_tr).tolist()}")

    smote_cache = None
    for name in models:
        strategies = (config.IMBALANCE_STRATEGIES if name in config.ABLATION_MODELS
                      else [config.MAIN_IMBALANCE_STRATEGY])
        for strategy in strategies:
            is_main = strategy == config.MAIN_IMBALANCE_STRATEGY
            X_fit, y_fit, smote_info = X_tr, y_tr, {}
            if strategy == "smote":
                if smote_cache is None:
                    smote_cache = apply_smote(X_tr, y_tr)
                    log.info(f"   SMOTE: {smote_cache[2]}")
                X_fit, y_fit, smote_info = smote_cache
            cw = "balanced" if strategy == "class_weight" else None

            log.info(f"[classical] {dataset}/{task}: {name} (imbalance={strategy})")
            est, val_f1, params, fit_time = tune_and_fit(name, X_fit, y_fit, X_va, y_va, cw)

            t0 = time.perf_counter()
            X_te_now = vec.transform(te["text_proc"])
            proba, pred = est.predict_proba(X_te_now), est.predict(X_te_now)
            latency_ms = 1000 * (time.perf_counter() - t0) / len(te)

            extra = {"imbalance_strategy": strategy, "best_params": params, "val_f1_macro": val_f1,
                     "train_time_s": fit_time, "inference_ms_per_post": latency_ms,
                     "n_train": int(len(y_fit)), "n_train_original": int(len(y_tr)),
                     "smote": smote_info, "n_features": int(X_tr.shape[1])}
            record_result(train_ds=dataset, eval_ds=dataset, task=task, model=name,
                          variant=strategy, is_main=is_main, y_true=y_te, proba=proba,
                          y_pred=pred, labels=labels, extra=extra)

            if is_main:
                path = utils.out_path("models", f"{dataset}__{task}__{name}.joblib")
                joblib.dump({"vectorizer": vec, "model": est, "labels": labels,
                             "input": "text_proc"}, path, compress=3)
                if task == "binary":
                    for other in cross_dataset_targets(dataset):
                        ote = utils.split_frames(utils.load_prepared(other))[2]
                        X_o = vec.transform(ote["text_proc"])
                        record_result(train_ds=dataset, eval_ds=other, task=task, model=name,
                                      variant=strategy, is_main=True,
                                      y_true=utils.encode_labels(ote, task),
                                      proba=est.predict_proba(X_o), y_pred=est.predict(X_o),
                                      labels=labels, extra=extra)


def run(datasets=None, models=None):
    utils.set_seed()
    models = models or config.CLASSICAL_MODELS
    for ds in datasets or utils.available_prepared():
        df = utils.load_prepared(ds)
        for task in config.TASKS[ds]:
            run_task(ds, task, df, models)
