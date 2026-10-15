"""Aggregates every result into tables, comparison figures, the deployment
model choice, and `findings_summary.md`, organised by research question."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import config
from src import utils
from src.statistical_tests import load_records

log = utils.get_logger()
METRICS = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted", "auc_roc"]


def md_table(df, floatfmt="{:.4f}"):
    if df is None or len(df) == 0:
        return "_No results available._\n"
    df = df.copy()
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        vals = []
        for v in row.values:
            if isinstance(v, (float, np.floating)):
                vals.append("" if np.isnan(v) else floatfmt.format(v))
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def _flat(records):
    keep = ["run_id", "train_dataset", "eval_dataset", "cross_dataset", "task", "model",
            "model_family", "variant", "is_main", "n_test"] + METRICS + [
        "f1_macro_ci_low", "f1_macro_ci_high", "precision_pos", "recall_pos", "f1_pos",
        "train_time_s", "inference_ms_per_post", "n_train", "val_f1_macro", "imbalance_strategy"]
    df = pd.DataFrame(records)
    for k in keep:
        if k not in df:
            df[k] = np.nan
    df = df[keep].copy()
    df["model_name"] = df["model"].map(config.MODEL_DISPLAY).fillna(df["model"])
    df["model_rank"] = df["model"].map({m: i for i, m in enumerate(config.MODEL_ORDER)})
    return df.sort_values(["train_dataset", "task", "eval_dataset", "model_rank", "variant"])


def _save_fig(fig, name):
    fig.tight_layout()
    fig.savefig(utils.out_path("figures", name), dpi=150)
    plt.close(fig)


def plot_main_comparison(main):
    main = main.assign(setting=main["train_dataset"] + " | " + main["task"])
    for metric in ["f1_macro", "auc_roc", "accuracy"]:
        fig, ax = plt.subplots(figsize=(11, 5))
        sns.barplot(data=main, x="setting", y=metric, hue="model_name", ax=ax,
                    hue_order=[config.MODEL_DISPLAY[m] for m in config.MODEL_ORDER
                               if config.MODEL_DISPLAY[m] in set(main["model_name"])])
        ax.set_ylim(max(0, main[metric].min() - 0.1), 1.0)
        ax.set_title(f"Model comparison ({metric}) - test set, identical conditions")
        ax.set_xlabel("")
        ax.legend(fontsize=8, ncol=3)
        _save_fig(fig, f"comparison_{metric}.png")

    piv = main.pivot_table(index="model_name", columns="setting", values="f1_macro")
    piv = piv.reindex([config.MODEL_DISPLAY[m] for m in config.MODEL_ORDER if config.MODEL_DISPLAY[m] in piv.index])
    fig, ax = plt.subplots(figsize=(1.8 * piv.shape[1] + 3, 0.6 * piv.shape[0] + 2))
    sns.heatmap(piv, annot=True, fmt=".3f", cmap="YlGnBu", ax=ax)
    ax.set_title("Macro F1 by model and dataset/task")
    _save_fig(fig, "heatmap_f1_macro.png")


def plot_per_class(records):
    rows = []
    for r in records:
        if not (r["is_main"] and not r["cross_dataset"]):
            continue
        for lab, m in r["per_class"].items():
            rows.append({"setting": f"{r['train_dataset']} | {r['task']}", "model": r["model"],
                         "model_name": config.MODEL_DISPLAY.get(r["model"], r["model"]),
                         "class": lab, "f1": m["f1"], "recall": m["recall"],
                         "precision": m["precision"], "support": m["support"]})
    pc = pd.DataFrame(rows)
    if pc.empty:
        return pc
    pc.to_csv(utils.out_path("tables", "per_class_metrics.csv"), index=False)
    for setting, g in pc.groupby("setting"):
        piv = g.pivot_table(index="model_name", columns="class", values="f1")
        piv = piv.reindex([config.MODEL_DISPLAY[m] for m in config.MODEL_ORDER if config.MODEL_DISPLAY[m] in piv.index])
        fig, ax = plt.subplots(figsize=(1.6 * piv.shape[1] + 3, 0.6 * piv.shape[0] + 2))
        sns.heatmap(piv, annot=True, fmt=".3f", cmap="RdYlGn", vmin=0, vmax=1, ax=ax)
        ax.set_title(f"Per-class F1 | {setting}")
        plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
        _save_fig(fig, f"per_class_f1__{setting.replace(' | ', '__')}.png")
    return pc


def cross_dataset_table(df):
    b = df[(df.task == "binary") & df.is_main]
    ind = b[~b.cross_dataset][["train_dataset", "model", "model_name", "f1_macro", "auc_roc"]]
    crs = b[b.cross_dataset][["train_dataset", "eval_dataset", "model", "f1_macro", "auc_roc"]]
    if crs.empty:
        return pd.DataFrame()
    t = crs.merge(ind, on=["train_dataset", "model"], suffixes=("_cross", "_in_domain"))
    t["f1_drop"] = t["f1_macro_in_domain"] - t["f1_macro_cross"]
    t["f1_drop_pct"] = 100 * t["f1_drop"] / t["f1_macro_in_domain"]
    t = t[["train_dataset", "eval_dataset", "model_name", "f1_macro_in_domain", "f1_macro_cross",
           "f1_drop", "f1_drop_pct", "auc_roc_in_domain", "auc_roc_cross", "model"]]
    t.to_csv(utils.out_path("tables", "cross_dataset_generalisation.csv"), index=False)

    t2 = t.assign(direction=t["train_dataset"] + " -> " + t["eval_dataset"])
    long = pd.concat([
        t2.assign(setting="in-domain", f1=t2["f1_macro_in_domain"]),
        t2.assign(setting="cross-dataset", f1=t2["f1_macro_cross"])])
    g = sns.catplot(data=long, x="model_name", y="f1", hue="setting", col="direction", kind="bar",
                    height=4.2, aspect=1.4)
    g.set_xticklabels(rotation=25, ha="right")
    g.figure.suptitle("Cross-dataset generalisation (binary task, macro F1)", y=1.03)
    g.figure.savefig(utils.out_path("figures", "cross_dataset_generalisation.png"), dpi=150,
                     bbox_inches="tight")
    plt.close(g.figure)
    return t


def imbalance_table(df):
    ab = df[(df.model_family == "classical") & ~df.cross_dataset].copy()
    if ab["variant"].nunique() < 2:
        return pd.DataFrame()
    recs = {r["run_id"]: r for r in load_records()}
    ab["minority_class_f1"] = ab["run_id"].map(
        lambda rid: min(v["f1"] for v in recs[rid]["per_class"].values()))
    t = ab[["train_dataset", "task", "model_name", "variant", "f1_macro", "recall_macro",
            "minority_class_f1", "accuracy", "auc_roc"]]
    t.to_csv(utils.out_path("tables", "class_imbalance_ablation.csv"), index=False)
    ab["setting"] = ab["train_dataset"] + " | " + ab["task"]
    g = sns.catplot(data=ab, x="model_name", y="f1_macro", hue="variant", col="setting",
                    kind="bar", col_wrap=3, height=3.6, aspect=1.2,
                    hue_order=[v for v in config.IMBALANCE_STRATEGIES if v in set(ab.variant)])
    g.set_xticklabels(rotation=20)
    g.figure.suptitle("Class-imbalance handling: none vs class weights vs SMOTE (macro F1)", y=1.02)
    g.figure.savefig(utils.out_path("figures", "class_imbalance_ablation.png"), dpi=150,
                     bbox_inches="tight")
    plt.close(g.figure)
    return t


def efficiency_plot(main):
    m = main.dropna(subset=["inference_ms_per_post"])
    if m.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=m, x="inference_ms_per_post", y="f1_macro", hue="model_name",
                    style="train_dataset", s=90, ax=ax)
    ax.set_xscale("log")
    ax.set_xlabel("Inference time per post (ms, log scale)")
    ax.set_title("Deployment trade-off: effectiveness vs inference cost")
    _save_fig(fig, "efficiency_tradeoff.png")


def choose_deployment_model(df):
    b = df[(df.task == "binary") & df.is_main]
    if b.empty:
        return {}
    rows = []
    for (ds, model), g in b.groupby(["train_dataset", "model"]):
        ind = g[~g.cross_dataset]
        if ind.empty:
            continue
        cross = g[g.cross_dataset]["f1_macro"].mean() if g.cross_dataset.any() else np.nan
        f_in = float(ind["f1_macro"].iloc[0])
        score = f_in if np.isnan(cross) else (f_in + cross) / 2
        family = config.MODEL_FAMILIES.get(model)
        if family == "classical":
            path = utils.out_path("models", f"{ds}__binary__{model}.joblib")
        else:
            path = utils.out_dir("models") / f"{ds}__binary__{model}"
        rows.append({"train_dataset": ds, "model": model, "family": family, "f1_in_domain": f_in,
                     "f1_cross_mean": cross, "selection_score": score,
                     "latency_ms": float(ind["inference_ms_per_post"].iloc[0]),
                     "artifact": str(path), "artifact_exists": path.exists()})
    t = pd.DataFrame(rows).sort_values("selection_score", ascending=False)
    t.to_csv(utils.out_path("tables", "deployment_candidates.csv"), index=False)
    avail = t[t.artifact_exists]
    if avail.empty:
        return {}
    best = avail.iloc[0].to_dict()
    light = avail[avail.family == "classical"]
    choice = {"recommended": best,
              "lightweight_alternative": light.iloc[0].to_dict() if len(light) else None,
              "selection_rule": "mean of in-domain and cross-dataset macro F1 on the binary task",
              "thresholds": {"flag_for_review": 0.5, "auto_hide": 0.85}}
    utils.save_json(choice, utils.out_path("models", "deployment_choice.json"))
    return choice


def write_summary(df, main, cross, imb, choice, hyp, chars):
    L = []
    L.append("# Findings summary - Cyberbullying detection study\n")
    if config.QUICK_MODE:
        L.append("> **QUICK MODE OUTPUT - smoke test only, NOT research findings.**\n")
    L.append("Auto-generated by `run_all.py`. All models share the same cleaned data, "
             "70/15/15 stratified split (seed %d) and metrics.\n" % config.SEED)

    L.append("## Datasets\n")
    if chars is not None:
        L.append(md_table(chars))

    L.append("## Main RQ - Which model performs most effectively?\n")
    cols = ["train_dataset", "task", "model_name"] + METRICS + ["f1_macro_ci_low", "f1_macro_ci_high"]
    L.append(md_table(main[cols]))
    best = main.loc[main.groupby(["train_dataset", "task"])["f1_macro"].idxmax(),
                    ["train_dataset", "task", "model_name", "f1_macro", "auc_roc"]]
    L.append("\n**Best model per dataset/task (macro F1):**\n\n" + md_table(best))
    overall = main.groupby("model_name")[["f1_macro", "auc_roc"]].mean().sort_values(
        "f1_macro", ascending=False).reset_index()
    L.append("\n**Mean over all dataset/task settings:**\n\n" + md_table(overall))
    if len(overall):
        L.append(f"\nOverall best: **{overall.iloc[0]['model_name']}** "
                 f"(mean macro F1 = {overall.iloc[0]['f1_macro']:.4f}).\n")

    L.append("\n## SQ1 - Literature\nAnswered in the literature review chapter (not computational).\n")

    L.append("\n## SQ2 - Classical ML vs deep learning vs transformers (H1a, H1b)\n")
    fam = main.groupby(["train_dataset", "task", "model_family"])["f1_macro"].max().unstack()
    L.append(md_table(fam.reset_index()))
    for k in ["H1a", "H1b_part1", "H1b_part2"]:
        if k in hyp:
            h = hyp[k]
            L.append(f"\n- **{k} ({h.get('hypothesis')})**: {h.get('decision', h.get('status'))} "
                     f"(mean F1 diff = {h.get('mean_f1_diff', float('nan')):.4f}; "
                     f"McNemar, Holm-corrected, alpha={config.ALPHA}).")
    fr = hyp.get("friedman_all_models", {})
    if fr:
        L.append(f"\n- Friedman test over {fr.get('blocks')} settings: chi2={fr.get('chi2', float('nan')):.3f}, "
                 f"p={fr.get('p_value', float('nan')):.4f}. Average rank (1 = best): {fr.get('average_rank')}.")
    L.append("\nDetails: `outputs/stats/pairwise_model_tests.csv`, "
             "`outputs/stats/H1a_metricwise_best_transformer_vs_best_classical.csv`.\n")

    L.append("\n## SQ3 - Performance across categories (explicit / implicit / hate speech) (H1c)\n")
    ei_path = utils.out_path("stats", "H1c_explicit_vs_implicit.csv")
    if ei_path.exists():
        ei = pd.read_csv(ei_path)
        if len(ei):
            ei["model"] = ei["model"].map(config.MODEL_DISPLAY)
            L.append(md_table(ei[["dataset", "model", "f1_not_cyberbullying", "f1_explicit",
                                  "f1_implicit", "f1_hate_speech", "recall_explicit",
                                  "recall_implicit", "p_value"]]))
    h = hyp.get("H1c", {})
    if h:
        L.append(f"\n- **H1c**: {h.get('decision', 'n/a')} (Wilcoxon p = {h.get('wilcoxon_p', float('nan')):.4f}; "
                 f"median F1 gap explicit-implicit = {h.get('median_f1_gap_explicit_minus_implicit', float('nan')):.4f}).")
    L.append("\nPer-class heatmaps: `outputs/figures/per_class_f1__*.png` (includes the Twitter "
             "native categories age/ethnicity/gender/religion/other).\n")

    L.append("\n## SQ4 - Impact of class imbalance and dataset characteristics\n")
    if imb is not None and len(imb):
        piv = imb.pivot_table(index=["train_dataset", "task", "model_name"], columns="variant",
                              values="f1_macro").reset_index()
        L.append("**Macro F1 by imbalance strategy:**\n\n" + md_table(piv))
        mpiv = imb.pivot_table(index=["train_dataset", "task", "model_name"], columns="variant",
                               values="minority_class_f1").reset_index()
        L.append("\n**Worst-class F1 by imbalance strategy:**\n\n" + md_table(mpiv))
    if cross is not None and len(cross):
        L.append("\n**Cross-dataset generalisation (binary):**\n\n" +
                 md_table(cross.drop(columns=["model"])))
    L.append("\nSee also `outputs/tables/eda_imbalance_summary.csv`, "
             "`outputs/error_analysis/f1_by_post_length.csv`, `f1_by_language_group.csv`.\n")

    L.append("\n## SQ5 - Operationalising the best model\n")
    if choice:
        r = choice["recommended"]
        L.append(f"- Recommended model: **{config.MODEL_DISPLAY.get(r['model'])}** trained on "
                 f"**{r['train_dataset']}** (in-domain F1={r['f1_in_domain']:.4f}, mean cross-dataset "
                 f"F1={r['f1_cross_mean']:.4f}, {r['latency_ms']:.2f} ms/post).")
        la = choice.get("lightweight_alternative")
        if la:
            L.append(f"- Lightweight alternative (CPU, low latency): **{config.MODEL_DISPLAY.get(la['model'])}** "
                     f"on {la['train_dataset']} (F1={la['f1_in_domain']:.4f}, {la['latency_ms']:.3f} ms/post).")
        L.append("- Framework: `deployment/app.py` (REST API) -> same pre-processing -> model -> "
                 "probability; >= 0.85 auto-hide, 0.50-0.85 human moderator review, < 0.50 allow. "
                 "Periodic re-training on newly labelled data (concept drift / new slang).")
    L.append("\nFull candidate list: `outputs/tables/deployment_candidates.csv`.\n")
    path = utils.out_path("tables", "findings_summary.md")
    path.write_text("\n".join(L), encoding="utf-8")
    (config.OUTPUT_DIR / "findings_summary.md").write_text("\n".join(L), encoding="utf-8")
    return path


def run():
    records = load_records()
    if not records:
        log.warning("[report] no results yet")
        return
    df = _flat(records)
    df.to_csv(utils.out_path("tables", "all_results.csv"), index=False)
    main = df[df.is_main & ~df.cross_dataset].copy()
    main.to_csv(utils.out_path("tables", "main_model_comparison.csv"), index=False)

    plot_main_comparison(main)
    plot_per_class(records)
    cross = cross_dataset_table(df)
    imb = imbalance_table(df)
    efficiency_plot(main)
    choice = choose_deployment_model(df)

    hyp_path = utils.out_path("stats", "hypothesis_results.json")
    hyp = utils.load_json(hyp_path) if hyp_path.exists() else {}
    ch_path = utils.out_path("tables", "eda_dataset_characteristics.csv")
    chars = pd.read_csv(ch_path) if ch_path.exists() else None
    path = write_summary(df, main, cross, imb, choice, hyp, chars)
    log.info(f"[report] summary written to {path}")
