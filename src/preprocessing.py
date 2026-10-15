"""Standardised pre-processing pipeline (proposal, 'Analysis of data'):

1. Text cleaning   - remove URLs, @mentions, IP addresses (POPIA anonymisation),
                     special characters, excessive whitespace; lower-case.
2. Deduplication   - hash-based (MD5 of cleaned text); duplicates with
                     conflicting labels are removed entirely.
3. Language ID     - langdetect on every post.
4. Tokenisation    - NLTK TweetTokenizer (classical + BiLSTM input);
                     BERT / AfriBERTa use their own sub-word tokenisers.
5. Stop-words      - NLTK English list (meaningful pronouns/negations kept),
                     followed by WordNet lemmatisation (normalisation).
6. Harmonisation   - binary + common 4-class labels.
7. Split           - stratified 70% train / 15% validation / 15% test.
SMOTE (class imbalance) is applied later, to the training split only.

Two text views are stored:
  text_clean : cleaned natural text   -> transformers (need full context)
  text_proc  : tokenised, stop-word-free, lemmatised -> TF-IDF models, BiLSTM
"""
import hashlib
import html
import re
from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import config
from src import utils
from src.data_loading import load_raw
from src.harmonisation import harmonise

log = utils.get_logger()

URL_RE = re.compile(r"(https?://\S+|www\.\S+|\S+\.(?:com|net|org|ly|co)/\S*)", re.I)
MENTION_RE = re.compile(r"@\w+")
IP_RE = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")
RT_RE = re.compile(r"\brt\b")
REPEAT_RE = re.compile(r"(.)\1{2,}")
SPECIAL_RE = re.compile(r"[^\w\s\.,!\?']|_")
WS_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    t = html.unescape(str(text))
    t = URL_RE.sub(" ", t)
    t = MENTION_RE.sub(" ", t)
    t = IP_RE.sub(" ", t)
    t = t.lower()
    t = RT_RE.sub(" ", t)
    t = t.replace("#", " ")
    t = REPEAT_RE.sub(r"\1\1", t)
    t = SPECIAL_RE.sub(" ", t)
    return WS_RE.sub(" ", t).strip()


# --------------------------------------------------------------------------
# NLTK resources
# --------------------------------------------------------------------------
def ensure_nltk():
    import nltk
    for res, path in [("stopwords", "corpora/stopwords"), ("wordnet", "corpora/wordnet"),
                      ("omw-1.4", "corpora/omw-1.4")]:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(res, quiet=True)


_TOKENIZER = None
_STOP = None
_LEMMATIZER = None


def _resources():
    global _TOKENIZER, _STOP, _LEMMATIZER
    if _TOKENIZER is None:
        ensure_nltk()
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer
        from nltk.tokenize import TweetTokenizer
        _TOKENIZER = TweetTokenizer(preserve_case=False, reduce_len=True)
        _STOP = set(stopwords.words("english")) - set(config.KEEP_STOPWORDS)
        _LEMMATIZER = WordNetLemmatizer()
    return _TOKENIZER, _STOP, _LEMMATIZER


@lru_cache(maxsize=500_000)
def _lemma(word: str) -> str:
    return _resources()[2].lemmatize(word)


def tokenize(text: str):
    tok, _, _ = _resources()
    return tok.tokenize(text)


def process_for_classical(text_clean: str) -> str:
    """Tokenise -> keep alphabetic tokens -> remove stop-words -> lemmatise."""
    _, stop, _ = _resources()
    tokens = [w for w in tokenize(text_clean) if w.isalpha() and len(w) > 1 and w not in stop]
    return " ".join(_lemma(w) for w in tokens)


# --------------------------------------------------------------------------
# Language identification
# --------------------------------------------------------------------------
def detect_languages(texts: pd.Series) -> pd.Series:
    from joblib import Parallel, delayed

    from src.langid import detect_batch
    texts = texts.tolist()
    chunks = [texts[i:i + 2000] for i in range(0, len(texts), 2000)]
    res = Parallel(n_jobs=-1)(delayed(detect_batch)(c, config.LANGDETECT_MIN_WORDS) for c in chunks)
    return pd.Series([lang for chunk in res for lang in chunk])


# --------------------------------------------------------------------------
# Full preparation pipeline
# --------------------------------------------------------------------------
def _stratified_sample(df, n, strat_col, seed):
    if n is None or len(df) <= n:
        return df
    sample, _ = train_test_split(df, train_size=n, stratify=df[strat_col], random_state=seed)
    return sample


def _safe_stratify(labels: pd.Series):
    """Stratify only if every class has at least 2 members."""
    return labels if labels.value_counts().min() >= 2 else None


def assign_splits(df: pd.DataFrame, strat_col: str, seed: int) -> pd.Series:
    idx = df.index.to_numpy()
    tr_idx, tmp_idx = train_test_split(
        idx, test_size=config.VAL_SIZE + config.TEST_SIZE,
        stratify=_safe_stratify(df[strat_col]), random_state=seed)
    rel_test = config.TEST_SIZE / (config.VAL_SIZE + config.TEST_SIZE)
    va_idx, te_idx = train_test_split(
        tmp_idx, test_size=rel_test,
        stratify=_safe_stratify(df.loc[tmp_idx, strat_col]), random_state=seed)
    split = pd.Series("train", index=df.index)
    split.loc[va_idx] = "val"
    split.loc[te_idx] = "test"
    return split


def prepare_dataset(dataset: str) -> pd.DataFrame:
    log.info(f"[prepare] {dataset}: loading raw data")
    df = load_raw(dataset)
    stats = {"dataset": dataset, "n_raw": len(df)}

    # 1. cleaning
    df["text_clean"] = df["text_raw"].map(clean_text)
    empty = df["text_clean"].str.len() == 0
    stats["n_empty_after_cleaning"] = int(empty.sum())
    df = df[~empty].copy()

    # 6. harmonisation (lexicon rule needs the cleaned text)
    df = harmonise(dataset, df)

    # 2. hash-based deduplication
    df["text_hash"] = df["text_clean"].map(lambda t: hashlib.md5(t.encode("utf-8")).hexdigest())
    fine_label = "label_native" if dataset == "twitter" else "label_h4"
    n_labels_per_hash = df.groupby("text_hash")[fine_label].transform("nunique")
    conflict = n_labels_per_hash > 1
    stats["n_conflicting_duplicates_removed"] = int(conflict.sum())
    df = df[~conflict]
    before = len(df)
    df = df.drop_duplicates("text_hash", keep="first")
    stats["n_exact_duplicates_removed"] = before - len(df)

    # optional equal-for-all-models subsample
    strat_col = "label_native" if dataset == "twitter" else "label_h4"
    df = _stratified_sample(df, config.MAX_ROWS, strat_col, config.SEED)

    # 4./5. tokenisation, stop-word removal, lemmatisation
    log.info(f"[prepare] {dataset}: tokenising / stop-words / lemmatising {len(df):,} texts")
    df["text_proc"] = df["text_clean"].map(process_for_classical)
    empty_proc = df["text_proc"].str.len() == 0
    stats["n_empty_after_stopwords_removed"] = int(empty_proc.sum())
    df = df[~empty_proc].reset_index(drop=True)

    # 3. language identification
    if config.RUN_LANGDETECT:
        log.info(f"[prepare] {dataset}: language identification")
        df["lang"] = detect_languages(df["text_clean"]).to_numpy()
    else:
        df["lang"] = "not_run"

    # descriptive features
    df["n_chars"] = df["text_raw"].str.len()
    df["n_tokens"] = df["text_clean"].str.split().str.len()
    df["n_tokens_proc"] = df["text_proc"].str.split().str.len()

    # 7. split
    df["split"] = assign_splits(df, strat_col, config.SEED).to_numpy()
    stats["n_final"] = len(df)
    stats.update({f"n_{s}": int((df["split"] == s).sum()) for s in ["train", "val", "test"]})

    df.to_pickle(utils.prepared_path(dataset))
    utils.save_json(stats, utils.out_path("data", f"{dataset}_preprocessing_stats.json"))
    log.info(f"[prepare] {dataset}: done -> {stats}")
    return df


def run(datasets=None):
    utils.set_seed()
    for ds in datasets or config.ACTIVE_DATASETS:
        prepare_dataset(ds)
