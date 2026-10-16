"""
Combines every finding produced by run_all.py into ONE long-format CSV:

    outputs/all_findings.csv

One row = one finding: section, research question, dataset, task, model, variant,
evaluation dataset, category, metric, value, detail.

    python export_findings.py                 # reads outputs/
    python export_findings.py --output-dir outputs_quick
"""
import argparse
import json
from pathlib import Path

import pandas as pd

import config

COLS = ["section", "research_question", "dataset", "task", "model", "variant",
        "eval_dataset", "category", "metric", "value", "detail"]
rows = []


def add(section, rq, metric, value=None, detail="", **kw):
    r = {c: "" for c in COLS}
    r.update(section=section, research_question=rq, metric=metric, detail=detail, **kw)
    r["value"] = "" if value is None or (isinstance(value, float) and pd.isna(value)) else value
    rows.append(r)


def read(base, rel):
    p = base / rel
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def name(m):
    return config.MODEL_DISPLAY.get(m, m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default=str(config.OUTPUT_DIR))
    base = Path(ap.parse_args().output_dir)

    # 1. Pre-processing
    for _, r in read(base, "tables/preprocessing_summary.csv").iterrows():
        for m, v in r.items():
            if m != "dataset":
                add("01_preprocessing", "Objective 2", m, v, dataset=r["dataset"])

    # 2. Dataset characteristics
    for _, r in read(base, "tables/eda_dataset_characteristics.csv").iterrows():
        for m, v in r.items():
            if m not in ("dataset", "title"):
                add("02_dataset_characteristics", "SQ4", m, v, dataset=r["dataset"], detail=r["title"])

    # 3. Class distribution and imbalance
    for _, r in read(base, "tables/eda_class_distribution.csv").iterrows():
        for m in ("count", "percent"):
            add("03_class_distribution", "SQ4", m, r[m], dataset=r["dataset"], task=r["task"], category=r["label"])
    for _, r in read(base, "tables/eda_imbalance_summary.csv").iterrows():
        for m in ("n_classes", "majority_share", "minority_share", "imbalance_ratio"):
            add("04_imbalance", "SQ4", m, r[m], dataset=r["dataset"], task=r["task"],
                detail=f"majority={r['majority_class']}; minority={r['minority_class']}")

    # 5. Main model comparison (in-domain test set)
    res = read(base, "tables/all_results.csv")
    metrics = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_macro_ci_low",
               "f1_macro_ci_high", "f1_weighted", "auc_roc", "precision_pos", "recall_pos", "f1_pos",
               "val_f1_macro", "train_time_s", "inference_ms_per_post", "n_train", "n_test"]
    if len(res):
        main_ = res[res.is_main & ~res.cross_dataset]
        for _, r in main_.iterrows():
            for m in metrics:
                add("05_model_performance", "Main RQ / SQ2", m, r[m], dataset=r["train_dataset"],
                    task=r["task"], model=name(r["model"]), variant=r["variant"],
                    eval_dataset=r["eval_dataset"], detail=r["model_family"])
        best = main_.loc[main_.groupby(["train_dataset", "task"])["f1_macro"].idxmax()]
        for _, r in best.iterrows():
            add("06_best_model", "Main RQ", "f1_macro", r["f1_macro"], dataset=r["train_dataset"],
                task=r["task"], model=name(r["model"]), detail="highest macro F1 for this dataset/task")
        for m, g in main_.groupby("model"):
            add("06_best_model", "Main RQ", "mean_f1_macro_all_settings", g["f1_macro"].mean(), model=name(m))
            add("06_best_model", "Main RQ", "mean_auc_roc_all_settings", g["auc_roc"].mean(), model=name(m))

    # 7. Per-class performance
    for _, r in read(base, "tables/per_class_metrics.csv").iterrows():
        ds, task = [s.strip() for s in r["setting"].split("|")]
        for m in ("precision", "recall", "f1", "support"):
            add("07_per_class", "SQ3", m, r[m], dataset=ds, task=task, model=r["model_name"], category=r["class"])

    # 8. Explicit vs implicit (H1c)
    for _, r in read(base, "stats/H1c_explicit_vs_implicit.csv").iterrows():
        for m in ("f1_explicit", "f1_implicit", "f1_hate_speech", "f1_not_cyberbullying",
                  "recall_explicit", "recall_implicit", "n_explicit", "n_implicit", "z", "p_value", "p_value_holm"):
            add("08_explicit_vs_implicit", "SQ3 / H1c", m, r[m], dataset=r["dataset"],
                task="harmonised4", model=name(r["model"]))

    # 9. Imbalance ablation
    for _, r in read(base, "tables/class_imbalance_ablation.csv").iterrows():
        for m in ("f1_macro", "recall_macro", "minority_class_f1", "accuracy", "auc_roc"):
            add("09_imbalance_ablation", "SQ4", m, r[m], dataset=r["train_dataset"], task=r["task"],
                model=r["model_name"], variant=r["variant"])

    # 10. Cross-dataset generalisation
    for _, r in read(base, "tables/cross_dataset_generalisation.csv").iterrows():
        for m in ("f1_macro_in_domain", "f1_macro_cross", "f1_drop", "f1_drop_pct", "auc_roc_in_domain", "auc_roc_cross"):
            add("10_cross_dataset", "SQ4", m, r[m], dataset=r["train_dataset"], task="binary",
                model=r["model_name"], eval_dataset=r["eval_dataset"])

    # 11. Pairwise significance tests
    for _, r in read(base, "stats/pairwise_model_tests.csv").iterrows():
        pair = f"{name(r['model_a'])} vs {name(r['model_b'])}"
        for m in ("f1_diff_a_minus_b", "diff_ci_low", "diff_ci_high", "bootstrap_p", "mcnemar_p",
                  "mcnemar_p_holm", "significant"):
            add("11_pairwise_tests", "SQ2 / H1b", m, r[m], dataset=r["dataset"], task=r["task"], model=pair)

    # 12. Hypotheses
    hp = base / "stats/hypothesis_results.json"
    if hp.exists():
        h = json.loads(hp.read_text(encoding="utf-8"))
        for key, d in h.items():
            if key == "friedman_all_models":
                add("12_hypotheses", "SQ2", "friedman_chi2", d.get("chi2"), category=key)
                add("12_hypotheses", "SQ2", "friedman_p_value", d.get("p_value"), category=key)
                for m, rank in d.get("average_rank", {}).items():
                    add("12_hypotheses", "SQ2", "average_rank", rank, model=name(m), category=key)
                continue
            for m, v in d.items():
                if m in ("hypothesis", "decision", "status"):
                    continue
                add("12_hypotheses", key, m, v, category=d.get("hypothesis", key))
            add("12_hypotheses", key, "decision", None, category=d.get("hypothesis", key),
                detail=d.get("decision") or ("Not tested - " + d.get("status", "")))

    # 13. Slices: post length and language
    for _, r in read(base, "error_analysis/f1_by_post_length.csv").iterrows():
        add("13_post_length", "SQ4", "f1_macro", r["f1_macro"], dataset=r["dataset"], task="binary",
            model=name(r["model"]), category=f"tokens {r['length_bucket']}", detail=f"n={r['n']}")
    for _, r in read(base, "error_analysis/f1_by_language_group.csv").iterrows():
        for m in ("accuracy", "f1_macro"):
            add("14_language", "SQ4", m, r[m], dataset=r["dataset"], task="binary",
                model=name(r["model"]), category=r["language_group"], detail=f"n={r['n']}")
    for _, r in read(base, "tables/eda_language_distribution.csv").iterrows():
        add("14_language", "SQ4", "percent_of_posts", r["percent"], dataset=r["dataset"],
            category=r["lang"], detail=f"n={r['count']}")

    # 15. Error analysis: most frequent confusions
    for p in sorted((base / "error_analysis").glob("confusions__*.csv")):
        _, ds, task, model = p.stem.split("__")
        for _, r in pd.read_csv(p).iterrows():
            add("15_confusions", "SQ3", "count", r["count"], dataset=ds, task=task, model=name(model),
                category=r["true -> predicted"])

    # 16. Feature importance (top 10 per model/class)
    fi = read(base, "error_analysis/feature_importance.csv")
    if len(fi):
        for (ds, task, model, cls), g in fi.groupby(["dataset", "task", "model", "class"]):
            for rank, (_, r) in enumerate(g.head(10).iterrows(), 1):
                add("16_feature_importance", "SQ3", "importance", r["importance"], dataset=ds, task=task,
                    model=name(model), category=cls, detail=f"rank {rank}: {r['feature']}")

    # 17. Deployment (SQ5)
    for _, r in read(base, "tables/deployment_candidates.csv").iterrows():
        for m in ("f1_in_domain", "f1_cross_mean", "selection_score", "latency_ms"):
            add("17_deployment", "SQ5", m, r[m], dataset=r["train_dataset"], task="binary", model=name(r["model"]))
    cp = base / "models/deployment_choice.json"
    if cp.exists():
        c = json.loads(cp.read_text(encoding="utf-8"))
        rec, light = c.get("recommended"), c.get("lightweight_alternative")
        if rec:
            add("17_deployment", "SQ5", "recommended_model", None, dataset=rec["train_dataset"],
                model=name(rec["model"]), detail=c.get("selection_rule", ""))
        if light:
            add("17_deployment", "SQ5", "lightweight_alternative", None, dataset=light["train_dataset"],
                model=name(light["model"]))
        for k, v in c.get("thresholds", {}).items():
            add("17_deployment", "SQ5", f"threshold_{k}", v)

    df = pd.DataFrame(rows, columns=COLS)
    df["value"] = df["value"].map(lambda v: round(v, 6) if isinstance(v, float) else v)
    out = base / "all_findings.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"{len(df):,} findings written to {out}")
    print(df.groupby("section").size().to_string())


if __name__ == "__main__":
    main()
