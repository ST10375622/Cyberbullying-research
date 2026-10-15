"""
Central configuration for the study:
"Natural Language Processing and Machine Learning Approaches for
 Cyberbullying Detection on Social Media" (L. Mosia, ST10375622).

Every model is trained and evaluated under the SAME conditions defined here
(same cleaned data, same 70/15/15 split, same seed, same metrics) so that
differences in performance can be attributed to model architecture.
"""
import os
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("CB_DATA_DIR", PROJECT_DIR.parent / "Datasets"))
OUTPUT_DIR = Path(os.environ.get("CB_OUTPUT_DIR", PROJECT_DIR / "outputs"))

# --------------------------------------------------------------------------
# Experimental conditions
# --------------------------------------------------------------------------
SEED = 42
TRAIN_SIZE, VAL_SIZE, TEST_SIZE = 0.70, 0.15, 0.15

DATASETS = {
    "twitter": {
        "file": "cyberbullying_tweets.csv",
        "title": "Twitter Cyberbullying Classification (Kaggle)",
    },
    "jigsaw": {
        "file": "train.csv",
        "title": "Jigsaw Toxic Comment Classification (Kaggle)",
    },
}
ACTIVE_DATASETS = ["twitter", "jigsaw"]

# Optional stratified subsample applied to a dataset BEFORE splitting, so that
# every model still sees identical data. None = use the full dataset.
MAX_ROWS = None

# Classification tasks run per dataset.
#   binary      : cyberbullying vs not cyberbullying
#   harmonised4 : common 4-class scheme (see Limitations section of proposal)
#   native6     : the Twitter dataset's own 6 categories
TASKS = {
    "twitter": ["binary", "harmonised4", "native6"],
    "jigsaw": ["binary", "harmonised4"],
}

BINARY_LABELS = ["not_cyberbullying", "cyberbullying"]
H4_LABELS = ["not_cyberbullying", "explicit_cyberbullying",
             "implicit_cyberbullying", "hate_speech"]
TWITTER_NATIVE_LABELS = ["not_cyberbullying", "age", "ethnicity", "gender",
                         "religion", "other_cyberbullying"]
JIGSAW_LABEL_COLS = ["toxic", "severe_toxic", "obscene", "threat",
                     "insult", "identity_hate"]

# --------------------------------------------------------------------------
# Label harmonisation (proposal, Limitations: "map dataset-specific categories
# onto a common four class scheme").
#   Twitter: identity-targeted categories -> hate_speech; the remaining
#            cyberbullying categories are split into explicit / implicit with
#            the explicit-abuse lexicon below.
#   Jigsaw : identity_hate -> hate_speech; obscene/insult/threat/severe_toxic
#            or a lexicon hit -> explicit; toxic only -> implicit.
# --------------------------------------------------------------------------
TWITTER_TO_H4 = {
    "not_cyberbullying": "not_cyberbullying",
    "religion": "hate_speech",
    "ethnicity": "hate_speech",
    "gender": "hate_speech",
    "age": "lexicon",
    "other_cyberbullying": "lexicon",
}

# Regex stems of overtly abusive / profane terms (matched at a word start).
# A post containing any of these is treated as EXPLICIT; abusive posts
# without them are treated as IMPLICIT (sarcasm, exclusion, mockery...).
EXPLICIT_LEXICON = [
    r"fuck", r"f\*ck", r"fck", r"shit", r"bitch", r"cunt", r"dick", r"cock",
    r"pussy", r"asshole", r"arsehole", r"ass\b", r"bastard", r"slut", r"whore",
    r"hoe\b", r"hoes\b", r"twat", r"wanker", r"prick", r"douche", r"motherf",
    r"idiot", r"stupid", r"moron", r"dumb", r"retard", r"imbecile", r"loser",
    r"ugly", r"fat\b", r"fatso", r"pig\b", r"scum", r"trash", r"garbage",
    r"pathetic", r"worthless", r"useless", r"disgusting", r"freak", r"creep",
    r"kill yourself", r"kys\b", r"go die", r"die\b", r"hang yourself",
    r"shut up", r"stfu", r"suck", r"hate you", r"i hate", r"piece of",
    r"damn", r"crap", r"jerk", r"clown", r"bully", r"bullied",
]

# --------------------------------------------------------------------------
# Pre-processing
# --------------------------------------------------------------------------
RUN_LANGDETECT = True          # language identification (langdetect)
LANGDETECT_MIN_WORDS = 3       # shorter texts are labelled "und"
# Stop-words that carry meaning for abuse detection are kept.
KEEP_STOPWORDS = {
    "not", "no", "nor", "don", "don't", "didn't", "isn't", "aren't", "won't",
    "you", "your", "yours", "yourself", "he", "she", "him", "her", "they",
    "them", "we", "i", "me", "my",
}

# --------------------------------------------------------------------------
# Classical ML (TF-IDF features)
# --------------------------------------------------------------------------
TFIDF_PARAMS = dict(ngram_range=(1, 2), max_features=50000, min_df=2,
                    max_df=0.95, sublinear_tf=True)
CLASSICAL_MODELS = ["logreg", "svm", "random_forest"]
# Small grids tuned on the VALIDATION split (selection metric: macro F1).
CLASSICAL_GRIDS = {
    "logreg": {"C": [0.5, 2.0, 8.0]},
    "svm": {"C": [0.1, 0.5, 1.0]},
    "random_forest": {"n_estimators": [200], "min_samples_leaf": [1, 3]},
}
# Class-imbalance handling (proposal: SMOTE). The ablation re-trains the
# listed models with each strategy to answer SQ4 (impact of class imbalance).
MAIN_IMBALANCE_STRATEGY = "smote"
IMBALANCE_STRATEGIES = ["none", "class_weight", "smote"]
# Random Forest is left out of the ablation by default because it is by far the
# slowest model on SMOTE-expanded Jigsaw data; add it if compute allows.
ABLATION_MODELS = ["logreg", "svm"]
# SMOTE raises every minority class to at least this fraction of the
# majority class (1.0 = fully balanced).
SMOTE_TARGET_RATIO = 0.5
SMOTE_K_NEIGHBORS = 5

# --------------------------------------------------------------------------
# Deep learning: BiLSTM (Keras 3; TensorFlow backend if installed, else torch)
# --------------------------------------------------------------------------
BILSTM = dict(max_vocab=50000, min_freq=2, max_len=100, embed_dim=128,
              lstm_units=64, dense_units=64, dropout=0.3, batch_size=128,
              epochs=8, patience=2, lr=1e-3)

# --------------------------------------------------------------------------
# Transformers (HuggingFace + PyTorch)
# --------------------------------------------------------------------------
TRANSFORMER_MODELS = {
    "bert": "bert-base-uncased",
    "afriberta": "castorini/afriberta_base",
}
TRANSFORMER = dict(max_len=128, batch_size=32, eval_batch_size=128, epochs=2,
                   lr=2e-5, weight_decay=0.01, warmup_ratio=0.1,
                   max_grad_norm=1.0, log_every=200)
# Save fine-tuned binary transformer models for deployment (~0.5 GB each).
SAVE_TRANSFORMER_BINARY_MODELS = True

# --------------------------------------------------------------------------
# Evaluation & statistics
# --------------------------------------------------------------------------
BOOTSTRAP_ITERATIONS = 1000
ALPHA = 0.05
N_MISCLASSIFIED_EXPORT = 50
MODEL_FAMILIES = {
    "logreg": "classical", "svm": "classical", "random_forest": "classical",
    "bilstm": "deep_learning", "bert": "transformer", "afriberta": "transformer",
}
MODEL_ORDER = ["logreg", "svm", "random_forest", "bilstm", "bert", "afriberta"]
MODEL_DISPLAY = {
    "logreg": "Logistic Regression", "svm": "SVM", "random_forest": "Random Forest",
    "bilstm": "BiLSTM", "bert": "BERT", "afriberta": "AfriBERTa",
}

QUICK_MODE = False


def apply_quick_mode():
    """Tiny smoke-test configuration: checks the pipeline end-to-end in minutes.
    Results produced in quick mode are NOT research findings."""
    global MAX_ROWS, BILSTM, TRANSFORMER, TRANSFORMER_MODELS, QUICK_MODE
    global BOOTSTRAP_ITERATIONS, CLASSICAL_GRIDS, OUTPUT_DIR, SAVE_TRANSFORMER_BINARY_MODELS
    QUICK_MODE = True
    MAX_ROWS = 3000
    OUTPUT_DIR = PROJECT_DIR / "outputs_quick"
    BILSTM = {**BILSTM, "epochs": 2, "max_len": 50}
    TRANSFORMER = {**TRANSFORMER, "epochs": 1, "max_len": 64, "log_every": 20}
    TRANSFORMER_MODELS = {"bert": "google/bert_uncased_L-2_H-128_A-2",
                          "afriberta": "google/bert_uncased_L-2_H-128_A-2"}
    SAVE_TRANSFORMER_BINARY_MODELS = True
    BOOTSTRAP_ITERATIONS = 200
    CLASSICAL_GRIDS = {"logreg": {"C": [2.0]}, "svm": {"C": [0.5]},
                       "random_forest": {"n_estimators": [50], "min_samples_leaf": [1]}}
