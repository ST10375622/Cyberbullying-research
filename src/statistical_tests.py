"""Statistical significance of model comparisons and hypothesis testing.

H1a/H0a  Transformers (BERT, AfriBERTa) vs classical ML (LR, SVM, RF)
H1b/H0b  BiLSTM vs classical ML (and transformers vs BiLSTM)
H1c/H0c  Performance on explicit vs implicit cyberbullying categories

Tests used
* McNemar's test (exact binomial when discordant pairs < 25) on paired test-set
  predictions of two models, Holm-Bonferroni corrected.
* Paired bootstrap 95% CI of the macro-F1 difference.
* Friedman test over all six models (blocks = dataset x task) + average ranks.
* Two-proportion z-test on recall (explicit vs implicit) per model, and a
  one-sided Wilcoxon signed-rank test over all model x dataset pairs.
"""
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

import config
from src import utils
from src.evaluation import macro_f1_fast

log = utils.get_logger()


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load_records():
    recs = [utils.load_json(p) for p in sorted(utils.out_dir("metrics").glob("*.json"))]
    return recs


def load_predictions(run_id):
    d = np.load(utils.out_path("predictions", f"{run_id}.npz"))
    return d["y_true"], d["y_pred"], d["proba"]


def main_in_domain(records):
    return [r for r in records if r["is_main"] and not r["cross_dataset"]]


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------
def mcnemar(y, a, b):
    ca, cb = a == y, b == y
    n01 = int(np.sum(ca & ~cb))   # a right, b wrong
    n10 = int(np.sum(~ca & cb))   # a wrong, b right
    n = n01 + n10
    if n == 0:
        return n01, n10, 0.0, 1.0
    if n < 25:
        p = stats.binomtest(min(n01, n10), n, 0.5).pvalue
        return n01, n10, float("nan"), float(p)
    chi2 = (abs(n01 - n10) - 1) ** 2 / n
    return n01, n10, float(chi2), float(stats.chi2.sf(chi2, 1))


def holm(pvals):
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def paired_bootstrap(y, a, b, n_classes, iters=None):
    iters = iters or config.BOOTSTRAP_ITERATIONS
    rng = np.random.default_rng(config.SEED)
    n = len(y)
    diffs = np.empty(iters)
    for i in range(iters):
        idx = rng.integers(0, n, n)
        diffs[i] = macro_f1_fast(y[idx], a[idx], n_classes) - macro_f1_fast(y[idx], b[idx], n_classes)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return float(lo), float(hi), float(min(p, 1.0))


def two_proportion_z(k1, n1, k2, n2):
    if n1 == 0 or n2 == 0:
        return float("nan"), float("nan")
    p1, p2 = k1 / n1, k2 / n2
    pool = (k1 + k2) / (n1 + n2)
    se = np.sqrt(pool * (1 - pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return 0.0, 1.0
    z = (p1 - p2) / se
    return float(z), float(2 * stats.norm.sf(abs(z)))


# --------------------------------------------------------------------------
# Analyses
# --------------------------------------------------------------------------
def pairwise_tests(records):
    groups = {}
    for r in main_in_domain(records):
        groups.setdefault((r["train_dataset"], r["task"]), {})[r["model"]] = r
    rows = []
    for (ds, task), models in groups.items():
        names = [m for m in config.MODEL_ORDER if m in models]
        preds = {m: load_predictions(models[m]["run_id"]) for m in names}
        n_classes = len(models[names[0]]["labels"])
        for a, b in combinations(names, 2):
            y, pa, _ = preds[a]
            _, pb, _ = preds[b]
            n01, n10, chi2, p = mcnemar(y, pa, pb)
            lo, hi, bp = paired_bootstrap(y, pa, pb, n_classes)
            fa, fb = models[a]["f1_macro"], models[b]["f1_macro"]
            rows.append({"dataset": ds, "task": task, "model_a": a, "model_b": b,
                         "family_a": config.MODEL_FAMILIES[a], "family_b": config.MODEL_FAMILIES[b],
                         "f1_macro_a": fa, "f1_macro_b": fb, "f1_diff_a_minus_b": fa - fb,
                         "diff_ci_low": lo, "diff_ci_high": hi, "bootstrap_p": bp,
                         "a_right_b_wrong": n01, "a_wrong_b_right": n10,
                         "mcnemar_chi2": chi2, "mcnemar_p": p})
    df = pd.DataFrame(rows)
    if len(df):
        df["mcnemar_p_holm"] = holm(df["mcnemar_p"].to_numpy())
        df["significant"] = df["mcnemar_p_holm"] < config.ALPHA
    return df


def _orient(df, fam_a, fam_b):
    """Rows comparing a model of family fam_a against fam_b, oriented a vs b."""
    out = []
    for _, r in df.iterrows():
        if r.family_a == fam_a and r.family_b == fam_b:
            out.append(r.to_dict())
        elif r.family_a == fam_b and r.family_b == fam_a:
            d = r.to_dict()
            d.update({"model_a": r.model_b, "model_b": r.model_a, "family_a": fam_a, "family_b": fam_b,
                      "f1_macro_a": r.f1_macro_b, "f1_macro_b": r.f1_macro_a,
                      "f1_diff_a_minus_b": -r.f1_diff_a_minus_b,
                      "diff_ci_low": -r.diff_ci_high, "diff_ci_high": -r.diff_ci_low,
                      "a_right_b_wrong": r.a_wrong_b_right, "a_wrong_b_right": r.a_right_b_wrong})
            out.append(d)
    return pd.DataFrame(out)


def summarise_family_test(df, fam_a, fam_b, name):
    sub = _orient(df, fam_a, fam_b)
    if sub.empty:
        return {"hypothesis": name, "status": "not enough models trained"}, sub
    better = int(((sub["f1_diff_a_minus_b"] > 0) & sub["significant"]).sum())
    worse = int(((sub["f1_diff_a_minus_b"] < 0) & sub["significant"]).sum())
    n = len(sub)
    if better == n:
        decision = f"Reject H0: {fam_a} significantly outperforms {fam_b} in all comparisons"
    elif better > n / 2:
        decision = f"Partial support for H1: {fam_a} significantly better in {better}/{n} comparisons"
    else:
        decision = f"Fail to reject H0: {fam_a} significantly better in only {better}/{n} comparisons"
    return {"hypothesis": name, "comparisons": n, "significantly_better": better,
            "significantly_worse": worse, "not_significant": n - better - worse,
            "mean_f1_diff": float(sub["f1_diff_a_minus_b"].mean()),
            "decision": decision}, sub


def metric_wise_family_comparison(records, fam_a, fam_b):
    """H1 states 'across ALL evaluation metrics': compare best model of each family."""
    metrics = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "auc_roc"]
    rows = []
    df = pd.DataFrame(main_in_domain(records))
    if df.empty:
        return pd.DataFrame()
    for (ds, task), g in df.groupby(["train_dataset", "task"]):
        a, b = g[g.model_family == fam_a], g[g.model_family == fam_b]
        if a.empty or b.empty:
            continue
        row = {"dataset": ds, "task": task}
        for m in metrics:
            row[f"{m}_best_{fam_a}"] = a[m].max()
            row[f"{m}_best_{fam_b}"] = b[m].max()
            row[f"{m}_{fam_a}_higher"] = bool(a[m].max() > b[m].max())
        rows.append(row)
    return pd.DataFrame(rows)


def friedman(records):
    df = pd.DataFrame(main_in_domain(records))
    if df.empty:
        return {}
    piv = df.pivot_table(index=["train_dataset", "task"], columns="model", values="f1_macro")
    piv = piv[[m for m in config.MODEL_ORDER if m in piv.columns]].dropna()
    res = {"blocks": len(piv), "models": list(piv.columns)}
    if piv.shape[1] >= 3 and piv.shape[0] >= 2:
        stat, p = stats.friedmanchisquare(*[piv[c].to_numpy() for c in piv.columns])
        res.update({"chi2": float(stat), "p_value": float(p)})
    ranks = piv.rank(axis=1, ascending=False).mean().sort_values()
    res["average_rank"] = ranks.round(3).to_dict()
    return res


def explicit_vs_implicit(records):
    rows = []
    for r in main_in_domain(records):
        if r["task"] != "harmonised4":
            continue
        y, pred, _ = load_predictions(r["run_id"])
        labels = r["labels"]
        e, i = labels.index("explicit_cyberbullying"), labels.index("implicit_cyberbullying")
        ke, ne = int(((y == e) & (pred == e)).sum()), int((y == e).sum())
        ki, ni = int(((y == i) & (pred == i)).sum()), int((y == i).sum())
        z, p = two_proportion_z(ke, ne, ki, ni)
        pc = r["per_class"]
        rows.append({"dataset": r["train_dataset"], "model": r["model"],
                     "f1_explicit": pc["explicit_cyberbullying"]["f1"],
                     "f1_implicit": pc["implicit_cyberbullying"]["f1"],
                     "f1_hate_speech": pc["hate_speech"]["f1"],
                     "f1_not_cyberbullying": pc["not_cyberbullying"]["f1"],
                     "recall_explicit": ke / ne if ne else np.nan,
                     "recall_implicit": ki / ni if ni else np.nan,
                     "n_explicit": ne, "n_implicit": ni, "z": z, "p_value": p})
    df = pd.DataFrame(rows)
    summary = {"hypothesis": "H1c: lower performance on implicit than explicit cyberbullying"}
    if len(df) >= 2:
        diff = df["f1_explicit"] - df["f1_implicit"]
        summary["pairs"] = int(len(df))
        summary["models_with_lower_implicit_f1"] = int((diff > 0).sum())
        summary["median_f1_gap_explicit_minus_implicit"] = float(diff.median())
        if (diff != 0).sum() >= 1:
            w = stats.wilcoxon(df["f1_explicit"], df["f1_implicit"], alternative="greater",
                               zero_method="zsplit")
            summary.update({"wilcoxon_stat": float(w.statistic), "wilcoxon_p": float(w.pvalue)})
            reject = w.pvalue < config.ALPHA
            summary["decision"] = ("Reject H0: implicit cyberbullying is detected significantly worse "
                                   "than explicit cyberbullying" if reject else
                                   "Fail to reject H0: no significant explicit/implicit difference")
        df["p_value_holm"] = holm(df["p_value"].fillna(1.0).to_numpy())
    return summary, df


def run():
    records = load_records()
    if not records:
        log.warning("[stats] no metric files found - train some models first")
        return {}
    log.info(f"[stats] {len(records)} result files")
    sdir = utils.out_dir("stats")

    pw = pairwise_tests(records)
    pw.to_csv(sdir / "pairwise_model_tests.csv", index=False)

    results = {}
    if len(pw):
        for fam_a, fam_b, key, name in [
            ("transformer", "classical", "H1a", "Transformers outperform classical ML"),
            ("deep_learning", "classical", "H1b_part1", "BiLSTM outperforms classical ML"),
            ("transformer", "deep_learning", "H1b_part2", "Transformers outperform BiLSTM"),
        ]:
            summ, sub = summarise_family_test(pw, fam_a, fam_b, name)
            results[key] = summ
            sub.to_csv(sdir / f"{key}_{fam_a}_vs_{fam_b}.csv", index=False)
    metric_wise_family_comparison(records, "transformer", "classical").to_csv(
        sdir / "H1a_metricwise_best_transformer_vs_best_classical.csv", index=False)
    metric_wise_family_comparison(records, "deep_learning", "classical").to_csv(
        sdir / "H1b_metricwise_bilstm_vs_best_classical.csv", index=False)

    results["friedman_all_models"] = friedman(records)
    summ, ei = explicit_vs_implicit(records)
    results["H1c"] = summ
    ei.to_csv(sdir / "H1c_explicit_vs_implicit.csv", index=False)

    utils.save_json(results, sdir / "hypothesis_results.json")
    log.info(f"[stats] hypothesis results: {results}")
    return results
