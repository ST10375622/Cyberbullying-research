"""
Evidence-based thresholds for the deployment framework (SQ5).

The screening model (Jigsaw-trained binary Logistic Regression, SMOTE) produces
probabilities that are not calibrated, so a raw score of 0.85 does not mean
"85% likely to be cyberbullying". This script:

1. splits the VALIDATION set 50/50 (stratified) into a calibration half and a
   threshold-selection half;
2. calibrates the frozen model on the calibration half (Platt / sigmoid);
3. on the threshold-selection half chooses
     - auto_hide  = lowest threshold whose precision >= TARGET_PRECISION
     - review     = highest threshold that still catches >= TARGET_RECALL of
                    all cyberbullying posts (hide + review);
4. reports on the untouched TEST set the share of posts auto-hidden, sent to
   human review and allowed, and the precision of the auto-hidden posts,
   alongside the original un-calibrated 0.85 / 0.50 thresholds for comparison.

    python calibrate_deployment.py
"""
import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import brier_score_loss, precision_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from src import utils  # noqa: E402

DATASET, MODEL = "jigsaw", "logreg"
TARGET_PRECISION = 0.95   # auto-hide only when at least 95% of hidden posts are abusive
TARGET_RECALL = 0.90      # hide + review together must catch at least 90% of abusive posts


def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    return float(sum(abs(y[idx == b].mean() - p[idx == b].mean()) * (idx == b).mean()
                     for b in range(bins) if (idx == b).any()))


def choose_thresholds(y, p):
    grid = np.unique(np.round(p, 4))
    n_above = np.array([(p >= t).sum() for t in grid])
    prec = np.array([y[p >= t].mean() if n else 1.0 for t, n in zip(grid, n_above)])
    # precision target must hold at this threshold and at every higher one
    stable = np.minimum.accumulate(prec[::-1])[::-1] >= TARGET_PRECISION
    cand = grid[stable & (n_above >= 20)]
    hide = float(cand.min()) if len(cand) else 1.0
    caught = np.array([(y[p >= t] == 1).sum() / max((y == 1).sum(), 1) for t in grid])
    ok = grid[caught >= TARGET_RECALL]
    review = float(ok.max()) if len(ok) else float(grid.min())
    return float(hide), min(review, float(hide))


def policy_report(y, p, hide, review, label):
    h, r = p >= hide, (p >= review) & (p < hide)
    a = ~(h | r)
    pos = max((y == 1).sum(), 1)
    return {
        "policy": label, "auto_hide_threshold": round(hide, 4), "review_threshold": round(review, 4),
        "share_auto_hidden": h.mean(), "share_human_review": r.mean(), "share_allowed": a.mean(),
        "precision_auto_hidden": y[h].mean() if h.any() else np.nan,
        "cyberbullying_share_in_review_queue": y[r].mean() if r.any() else np.nan,
        "share_of_cyberbullying_auto_hidden": (y[h] == 1).sum() / pos,
        "share_of_cyberbullying_reviewed": (y[r] == 1).sum() / pos,
        "share_of_cyberbullying_missed": (y[a] == 1).sum() / pos,
        "false_auto_hides_per_10k_posts": 10_000 * ((y == 0) & h).sum() / len(y),
    }


def main():
    df = utils.load_prepared(DATASET)
    _, va, te = utils.split_frames(df)
    bundle = joblib.load(utils.out_path("models", f"{DATASET}__binary__{MODEL}.joblib"))
    vec, model = bundle["vectorizer"], bundle["model"]
    y_va, y_te = utils.encode_labels(va, "binary"), utils.encode_labels(te, "binary")
    X_va, X_te = vec.transform(va["text_proc"]), vec.transform(te["text_proc"])

    i_cal, i_thr = train_test_split(np.arange(len(y_va)), test_size=0.5, stratify=y_va,
                                    random_state=config.SEED)
    cal = CalibratedClassifierCV(FrozenEstimator(model), method="sigmoid").fit(X_va[i_cal], y_va[i_cal])

    raw_te = model.predict_proba(X_te)[:, 1]
    cal_thr = cal.predict_proba(X_va[i_thr])[:, 1]
    cal_te = cal.predict_proba(X_te)[:, 1]
    hide, review = choose_thresholds(y_va[i_thr], cal_thr)

    calib = pd.DataFrame([
        {"probabilities": "raw (SMOTE-trained)", "brier_score": brier_score_loss(y_te, raw_te), "ece": ece(y_te, raw_te),
         "mean_predicted": raw_te.mean(), "observed_rate": y_te.mean()},
        {"probabilities": "calibrated (sigmoid on validation)", "brier_score": brier_score_loss(y_te, cal_te),
         "ece": ece(y_te, cal_te), "mean_predicted": cal_te.mean(), "observed_rate": y_te.mean()},
    ])
    pol = pd.DataFrame([
        policy_report(y_te, raw_te, 0.85, 0.50, "original: raw scores, 0.85 / 0.50 (not justified)"),
        policy_report(y_te, cal_te, hide, review,
                      f"evidence-based: calibrated, precision>={TARGET_PRECISION:.2f} / recall>={TARGET_RECALL:.2f}"),
    ])
    calib.to_csv(utils.out_path("tables", "deployment_calibration.csv"), index=False)
    pol.to_csv(utils.out_path("tables", "deployment_thresholds.csv"), index=False)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    for p, lab in [(raw_te, "raw"), (cal_te, "calibrated")]:
        fr, mp = calibration_curve(y_te, p, n_bins=10, strategy="quantile")
        ax.plot(mp, fr, marker="o", label=lab)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("Predicted probability of cyberbullying")
    ax.set_ylabel("Observed share of cyberbullying")
    ax.set_title("Reliability of screening-model scores (Jigsaw test set)", fontsize=10)
    ax.legend()
    fig.tight_layout()
    fig.savefig(utils.out_path("figures", "deployment_reliability.png"), dpi=150)
    plt.close(fig)

    joblib.dump({"vectorizer": vec, "model": cal, "labels": bundle["labels"], "input": "text_proc",
                 "thresholds": {"auto_hide": hide, "flag_for_review": review}},
                utils.out_path("models", f"{DATASET}__binary__{MODEL}__calibrated.joblib"), compress=3)
    utils.save_json({"model": f"{DATASET}__binary__{MODEL}__calibrated",
                     "thresholds": {"auto_hide": hide, "flag_for_review": review},
                     "target_precision": TARGET_PRECISION, "target_recall": TARGET_RECALL},
                    utils.out_path("models", "deployment_calibrated.json"))
    pd.set_option("display.width", 200)
    print(calib.round(4).to_string(index=False))
    print(pol.round(4).T.to_string())


if __name__ == "__main__":
    main()
