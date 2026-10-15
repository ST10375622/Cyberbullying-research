"""Label harmonisation onto the common scheme described in the proposal:
not cyberbullying / explicit cyberbullying / implicit cyberbullying / hate speech.
"""
import re

import numpy as np
import pandas as pd

import config

_LEXICON_RE = None


def lexicon_regex():
    global _LEXICON_RE
    if _LEXICON_RE is None:
        _LEXICON_RE = re.compile(r"\b(?:" + "|".join(config.EXPLICIT_LEXICON) + r")",
                                 flags=re.IGNORECASE)
    return _LEXICON_RE


def has_explicit_term(texts: pd.Series) -> pd.Series:
    rx = lexicon_regex()
    return texts.fillna("").map(lambda t: bool(rx.search(t)))


def harmonise_twitter(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["label_binary"] = np.where(df["label_native"] == "not_cyberbullying",
                                  "not_cyberbullying", "cyberbullying")
    explicit = has_explicit_term(df["text_clean"])
    mapped = df["label_native"].map(config.TWITTER_TO_H4)
    if mapped.isna().any():
        raise ValueError(f"Unmapped Twitter labels: {df.loc[mapped.isna(), 'label_native'].unique()}")
    lex = mapped == "lexicon"
    mapped = mapped.where(~lex, np.where(explicit, "explicit_cyberbullying",
                                         "implicit_cyberbullying"))
    df["label_h4"] = mapped
    df["has_explicit_term"] = explicit
    return df


def harmonise_jigsaw(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    lab = df[config.JIGSAW_LABEL_COLS]
    any_toxic = lab.max(axis=1) == 1
    df["label_binary"] = np.where(any_toxic, "cyberbullying", "not_cyberbullying")
    explicit_terms = has_explicit_term(df["text_clean"])
    explicit_labels = lab[["obscene", "insult", "threat", "severe_toxic"]].max(axis=1) == 1
    h4 = np.select(
        [~any_toxic, lab["identity_hate"] == 1, explicit_labels | explicit_terms],
        ["not_cyberbullying", "hate_speech", "explicit_cyberbullying"],
        default="implicit_cyberbullying")
    df["label_h4"] = h4
    df["label_native"] = np.where(any_toxic, "toxic_any", "clean")  # descriptive only
    df["has_explicit_term"] = explicit_terms
    return df


def harmonise(dataset: str, df: pd.DataFrame) -> pd.DataFrame:
    return {"twitter": harmonise_twitter, "jigsaw": harmonise_jigsaw}[dataset](df)


def harmonisation_table() -> pd.DataFrame:
    """Documentation table of the mapping rules (for the methodology chapter)."""
    rows = []
    for src, dst in config.TWITTER_TO_H4.items():
        rule = ("explicit if explicit-abuse lexicon match else implicit"
                if dst == "lexicon" else dst)
        rows.append({"dataset": "twitter", "source_label": src, "harmonised_label": rule})
    rows += [
        {"dataset": "jigsaw", "source_label": "all six labels = 0", "harmonised_label": "not_cyberbullying"},
        {"dataset": "jigsaw", "source_label": "identity_hate = 1", "harmonised_label": "hate_speech"},
        {"dataset": "jigsaw", "source_label": "obscene/insult/threat/severe_toxic = 1 or lexicon match",
         "harmonised_label": "explicit_cyberbullying"},
        {"dataset": "jigsaw", "source_label": "toxic = 1 only, no lexicon match",
         "harmonised_label": "implicit_cyberbullying"},
    ]
    return pd.DataFrame(rows)
