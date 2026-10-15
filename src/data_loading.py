"""Loading of the two publicly available Kaggle datasets (secondary data)."""
import pandas as pd

import config


def load_raw(dataset: str) -> pd.DataFrame:
    """Return the raw dataset with a common `text_raw` column.

    twitter -> columns: text_raw, label_native
    jigsaw  -> columns: text_raw, toxic, severe_toxic, obscene, threat,
               insult, identity_hate
    """
    path = config.DATA_DIR / config.DATASETS[dataset]["file"]
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {path}\n"
            f"Set CB_DATA_DIR or pass --data-dir to point at the Datasets folder.")

    if dataset == "twitter":
        df = pd.read_csv(path, encoding="utf-8")
        df = df.rename(columns={"tweet_text": "text_raw",
                                "cyberbullying_type": "label_native"})
        df = df[["text_raw", "label_native"]]
        df["label_native"] = df["label_native"].str.strip()
    elif dataset == "jigsaw":
        df = pd.read_csv(path, encoding="utf-8")
        df = df.rename(columns={"comment_text": "text_raw"})
        df = df[["text_raw"] + config.JIGSAW_LABEL_COLS]
        df[config.JIGSAW_LABEL_COLS] = df[config.JIGSAW_LABEL_COLS].astype(int)
    else:
        raise ValueError(f"Unknown dataset {dataset}")

    df = df.dropna(subset=["text_raw"]).reset_index(drop=True)
    df["text_raw"] = df["text_raw"].astype(str)
    df.insert(0, "dataset", dataset)
    return df
