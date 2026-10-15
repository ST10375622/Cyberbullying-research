"""Post-hoc analyses:
* Most predictive n-grams (Logistic Regression coefficients, Random Forest
  importances) - proposal: "feature importance scores that will identify which
  phrases are most predictive of cyberbullying".
* Misclassified posts of the best model per task (for manual examination;
  NOTE: contains offensive content - see ethical considerations).
* Performance by post length (dataset characteristic, SQ4).
* Performance on English vs non-English posts (language bias gap).
"""
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import f1_score

import config
from src import utils
from src.statistical_tests import load_predictions, load_records, main_in_domain

log = utils.get_logger()


def _feature_names(vec):
    return np.asarray(vec.get_feature_names_out())


def feature_importance(top_n=20):
    rows = []
    for path in sorted(utils.out_dir("models").glob("*.joblib")):
        ds, task, model = path.stem.split("__")
        if model not in ("logreg", "random_forest"):
            continue
        bundle = joblib.load(path)
        names = _feature_names(bundle["vectorizer"])
        labels = bundle["labels"]
        est = bundle["model"]
        if model == "logreg":
            coef = est.coef_
            if coef.shape[0] == 1:  # binary: positive weights -> cyberbullying
                coef = np.vstack([-coef[0], coef[0]])
            for ci, lab in enumerate(labels):
                if lab == "not_cyberbullying":
                    continue
                top = np.argsort(coef[ci])[::-1][:top_n]
                rows += [{"dataset": ds, "task": task, "model": model, "class": lab,
                          "feature": names[j], "importance": float(coef[ci, j])} for j in top]
        else:
            imp = est.feature_importances_
            top = np.argsort(imp)[::-1][:top_n]
            rows += [{"dataset": ds, "task": task, "model": model, "class": "all",
                      "feature": names[j], "importance": float(imp[j])} for j in top]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df.to_csv(utils.out_path("error_analysis", "feature_importance.csv"), index=False)

    for (ds, task, model), g in df.groupby(["dataset", "task", "model"]):
        classes = g["class"].unique()
        fig, axes = plt.subplots(1, len(classes), figsize=(4.5 * len(classes), 6))
        axes = np.atleast_1d(axes)
        for ax, c in zip(axes, classes):
            sub = g[g["class"] == c].head(15)
            sns.barplot(data=sub, x="importance", y="feature", ax=ax, color="darkorange")
            ax.set_title(c, fontsize=9)
            ax.set_ylabel("")
        fig.suptitle(f"Most predictive n-grams | {config.MODEL_DISPLAY[model]} | {ds} {task}")
        fig.tight_layout()
        fig.savefig(utils.out_path("error_analysis", f"feature_importance__{ds}__{task}__{model}.png"),
                    dpi=150)
        plt.close(fig)
    return df


def _best_by_task(records):
    best = {}
    for r in main_in_domain(records):
        key = (r["train_dataset"], r["task"])
        if key not in best or r["f1_macro"] > best[key]["f1_macro"]:
            best[key] = r
    return best


def misclassified_examples(records):
    out = []
    for (ds, task), r in _best_by_task(records).items():
        te = utils.split_frames(utils.load_prepared(ds))[2].reset_index(drop=True)
        y, pred, proba = load_predictions(r["run_id"])
        labels = r["labels"]
        wrong = np.where(y != pred)[0]
        conf = proba[wrong, pred[wrong]]
        order = wrong[np.argsort(-conf)][:config.N_MISCLASSIFIED_EXPORT]
        for i in order:
            out.append({"dataset": ds, "task": task, "model": r["model"],
                        "true_label": labels[y[i]], "predicted_label": labels[pred[i]],
                        "confidence": float(proba[i, pred[i]]), "lang": te.loc[i, "lang"],
                        "text": te.loc[i, "text_clean"][:400]})
        # most frequent confusions
        conf_pairs = pd.Series([f"{labels[a]} -> {labels[b]}" for a, b in zip(y[wrong], pred[wrong])])
        conf_pairs.value_counts().rename_axis("true -> predicted").reset_index(name="count").to_csv(
            utils.out_path("error_analysis", f"confusions__{ds}__{task}__{r['model']}.csv"), index=False)
    df = pd.DataFrame(out)
    df.to_csv(utils.out_path("error_analysis", "misclassified_examples_best_models.csv"),
              index=False, encoding="utf-8-sig")
    return df


def performance_by_slice(records):
    """Macro F1 per post-length quartile and per language group (binary task)."""
    len_rows, lang_rows = [], []
    for r in main_in_domain(records):
        if r["task"] != "binary":
            continue
        ds = r["train_dataset"]
        te = utils.split_frames(utils.load_prepared(ds))[2].reset_index(drop=True)
        y, pred, _ = load_predictions(r["run_id"])
        buckets = pd.qcut(te["n_tokens"], 4, duplicates="drop")
        for b in buckets.cat.categories:
            m = (buckets == b).to_numpy()
            len_rows.append({"dataset": ds, "model": r["model"], "length_bucket": str(b),
                             "n": int(m.sum()),
                             "f1_macro": f1_score(y[m], pred[m], average="macro", zero_division=0)})
        lang_group = np.where(te["lang"] == "en", "english", "non_english/undetermined")
        for g in np.unique(lang_group):
            m = lang_group == g
            lang_rows.append({"dataset": ds, "model": r["model"], "language_group": g,
                              "n": int(m.sum()), "accuracy": float((y[m] == pred[m]).mean()),
                              "f1_macro": f1_score(y[m], pred[m], average="macro", zero_division=0)})
    ld, lg = pd.DataFrame(len_rows), pd.DataFrame(lang_rows)
    ld.to_csv(utils.out_path("error_analysis", "f1_by_post_length.csv"), index=False)
    lg.to_csv(utils.out_path("error_analysis", "f1_by_language_group.csv"), index=False)
    if not ld.empty:
        for ds, g in ld.groupby("dataset"):
            fig, ax = plt.subplots(figsize=(8, 4.5))
            g = g.assign(model=g["model"].map(config.MODEL_DISPLAY))
            sns.pointplot(data=g, x="length_bucket", y="f1_macro", hue="model", ax=ax)
            ax.set_title(f"{ds}: binary macro F1 by post length (tokens, quartiles)")
            ax.set_xlabel("Post length bucket (tokens)")
            fig.tight_layout()
            fig.savefig(utils.out_path("figures", f"f1_by_length__{ds}.png"), dpi=150)
            plt.close(fig)
    return ld, lg


def run():
    records = load_records()
    if not records:
        log.warning("[analysis] no results yet")
        return
    log.info("[analysis] feature importance")
    feature_importance()
    log.info("[analysis] misclassified examples")
    misclassified_examples(records)
    log.info("[analysis] performance by length / language")
    performance_by_slice(records)
