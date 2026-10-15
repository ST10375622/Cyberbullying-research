"""Exploratory data analysis and dataset characterisation (supports SQ4:
'What impact do class imbalances and dataset characteristics have...')."""
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import config
from src import utils
from src.harmonisation import harmonisation_table

log = utils.get_logger()
sns.set_theme(style="whitegrid")


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(utils.out_path("eda", name), dpi=150)
    plt.close(fig)


def _bar_counts(series, title, fname, order=None):
    counts = series.value_counts()
    if order is not None:
        counts = counts.reindex([o for o in order if o in counts.index])
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=counts.values, y=counts.index.astype(str), ax=ax, color="steelblue")
    for i, v in enumerate(counts.values):
        ax.text(v, i, f" {v:,} ({v / counts.sum():.1%})", va="center", fontsize=8)
    ax.set_title(title)
    ax.set_xlabel("Number of posts")
    _save(fig, fname)
    return counts


def class_distribution_table(df, dataset):
    rows = []
    for task in config.TASKS[dataset]:
        col = utils.task_label_column(task)
        vc = df[col].value_counts()
        for lab, n in vc.items():
            rows.append({"dataset": dataset, "task": task, "label": lab, "count": int(n),
                         "percent": round(100 * n / len(df), 2)})
    return pd.DataFrame(rows)


def imbalance_summary(df, dataset):
    rows = []
    for task in config.TASKS[dataset]:
        vc = df[utils.task_label_column(task)].value_counts()
        rows.append({"dataset": dataset, "task": task, "n_classes": len(vc),
                     "majority_class": vc.index[0], "majority_share": round(vc.iloc[0] / vc.sum(), 4),
                     "minority_class": vc.index[-1], "minority_share": round(vc.iloc[-1] / vc.sum(), 4),
                     "imbalance_ratio": round(vc.iloc[0] / vc.iloc[-1], 2)})
    return pd.DataFrame(rows)


def split_distribution(df, dataset):
    rows = []
    for task in config.TASKS[dataset]:
        col = utils.task_label_column(task)
        ct = pd.crosstab(df["split"], df[col], normalize="index").round(4)
        ct["n"] = df["split"].value_counts()
        ct.insert(0, "task", task)
        ct.insert(0, "dataset", dataset)
        rows.append(ct.reset_index())
    return pd.concat(rows, ignore_index=True)


def dataset_characteristics(df, dataset, stats):
    tokens = df["text_proc"].str.split()
    vocab = Counter(w for toks in tokens for w in toks)
    n_tokens_total = int(df["n_tokens_proc"].sum())
    lang_share_en = (df["lang"] == "en").mean() if "lang" in df else np.nan
    return {
        "dataset": dataset,
        "title": config.DATASETS[dataset]["title"],
        "n_raw": stats.get("n_raw"),
        "n_after_cleaning": len(df),
        "duplicates_removed": stats.get("n_exact_duplicates_removed", 0)
        + stats.get("n_conflicting_duplicates_removed", 0),
        "cyberbullying_share": round((df["label_binary"] == "cyberbullying").mean(), 4),
        "mean_tokens": round(df["n_tokens"].mean(), 2),
        "median_tokens": float(df["n_tokens"].median()),
        "p95_tokens": float(df["n_tokens"].quantile(0.95)),
        "mean_chars": round(df["n_chars"].mean(), 1),
        "vocabulary_size": len(vocab),
        "type_token_ratio": round(len(vocab) / max(n_tokens_total, 1), 4),
        "share_english": round(lang_share_en, 4),
        "n_languages_detected": int(df["lang"].nunique()),
        "explicit_term_share": round(df["has_explicit_term"].mean(), 4),
    }


def top_ngrams(df, col_label, n=15, ngram=1):
    rows = []
    for lab, grp in df.groupby(col_label):
        c = Counter()
        for t in grp["text_proc"]:
            toks = t.split()
            grams = zip(*[toks[i:] for i in range(ngram)])
            c.update(" ".join(g) for g in grams)
        for term, cnt in c.most_common(n):
            rows.append({"label": lab, "ngram": term, "count": cnt})
    return pd.DataFrame(rows)


def plot_lengths(df, dataset):
    clip = df["n_tokens"].quantile(0.99)
    d = df[df["n_tokens"] <= clip]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    sns.histplot(data=d, x="n_tokens", hue="label_binary", bins=50, element="step",
                 stat="density", common_norm=False, ax=axes[0])
    axes[0].set_title(f"{dataset}: post length by binary label (<= p99)")
    sns.boxplot(data=d, y="label_h4", x="n_tokens", ax=axes[1], order=config.H4_LABELS,
                color="lightsteelblue", showfliers=False)
    axes[1].set_title(f"{dataset}: post length by harmonised class")
    _save(fig, f"{dataset}_length_distribution.png")


def plot_ngrams(ng, dataset, ngram):
    labs = ng["label"].unique()
    fig, axes = plt.subplots(1, len(labs), figsize=(4.2 * len(labs), 5))
    axes = np.atleast_1d(axes)
    for ax, lab in zip(axes, labs):
        sub = ng[ng["label"] == lab]
        sns.barplot(data=sub, x="count", y="ngram", ax=ax, color="indianred")
        ax.set_title(lab, fontsize=9)
        ax.set_ylabel("")
    fig.suptitle(f"{dataset}: top {'uni' if ngram == 1 else 'bi'}grams per harmonised class")
    _save(fig, f"{dataset}_top_{ngram}grams.png")


def run(datasets=None):
    datasets = datasets or utils.available_prepared()
    all_dist, all_imb, all_split, chars, lang_rows = [], [], [], [], []
    for ds in datasets:
        log.info(f"[eda] {ds}")
        df = utils.load_prepared(ds)
        stats = utils.load_json(utils.out_path("data", f"{ds}_preprocessing_stats.json"))

        all_dist.append(class_distribution_table(df, ds))
        all_imb.append(imbalance_summary(df, ds))
        all_split.append(split_distribution(df, ds))
        chars.append(dataset_characteristics(df, ds, stats))

        if ds == "twitter":
            _bar_counts(df["label_native"], "Twitter: original 6 categories",
                        f"{ds}_native_distribution.png", config.TWITTER_NATIVE_LABELS)
        else:
            prev = df[config.JIGSAW_LABEL_COLS].mean().sort_values(ascending=False)
            fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
            sns.barplot(x=prev.values * 100, y=prev.index, ax=axes[0], color="steelblue")
            axes[0].set_xlabel("% of comments")
            axes[0].set_title("Jigsaw: prevalence of each (multi-)label")
            corr = df[config.JIGSAW_LABEL_COLS].corr()
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="Blues", ax=axes[1])
            axes[1].set_title("Jigsaw: label co-occurrence (correlation)")
            _save(fig, f"{ds}_native_labels.png")
        _bar_counts(df["label_binary"], f"{ds}: binary labels", f"{ds}_binary_distribution.png",
                    config.BINARY_LABELS)
        _bar_counts(df["label_h4"], f"{ds}: harmonised 4-class labels",
                    f"{ds}_harmonised4_distribution.png", config.H4_LABELS)
        if ds == "twitter":
            ct = pd.crosstab(df["label_native"], df["label_h4"])
            ct.to_csv(utils.out_path("eda", "twitter_native_vs_harmonised.csv"))

        plot_lengths(df, ds)
        df.groupby("label_h4")[["n_tokens", "n_chars"]].describe().round(2).to_csv(
            utils.out_path("eda", f"{ds}_length_stats_by_class.csv"))

        lang = df["lang"].value_counts()
        for l, n in lang.items():
            lang_rows.append({"dataset": ds, "lang": l, "count": int(n),
                              "percent": round(100 * n / len(df), 2)})
        _bar_counts(df["lang"].where(df["lang"].isin(lang.index[:12]), "other"),
                    f"{ds}: detected languages (langdetect)", f"{ds}_languages.png")

        for ngram in (1, 2):
            ng = top_ngrams(df, "label_h4", n=15, ngram=ngram)
            ng.to_csv(utils.out_path("eda", f"{ds}_top_{ngram}grams.csv"), index=False)
            plot_ngrams(ng, ds, ngram)

    pd.concat(all_dist).to_csv(utils.out_path("tables", "eda_class_distribution.csv"), index=False)
    pd.concat(all_imb).to_csv(utils.out_path("tables", "eda_imbalance_summary.csv"), index=False)
    pd.concat(all_split).to_csv(utils.out_path("tables", "eda_split_distribution.csv"), index=False)
    pd.DataFrame(chars).to_csv(utils.out_path("tables", "eda_dataset_characteristics.csv"), index=False)
    pd.DataFrame(lang_rows).to_csv(utils.out_path("tables", "eda_language_distribution.csv"), index=False)
    harmonisation_table().to_csv(utils.out_path("tables", "label_harmonisation_rules.csv"), index=False)
    prep = [utils.load_json(utils.out_path("data", f"{d}_preprocessing_stats.json")) for d in datasets]
    pd.DataFrame(prep).to_csv(utils.out_path("tables", "preprocessing_summary.csv"), index=False)
    log.info("[eda] done")
