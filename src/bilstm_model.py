"""Model 4: Bidirectional LSTM (Keras 3).

Input: the same tokenised / stop-word-free / lemmatised text as the classical
models. SMOTE is not meaningful for raw token sequences, so class imbalance is
handled with balanced class weights in the loss. The epoch with the best
validation macro F1 is kept (early stopping).

Keras 3 runs on TensorFlow when it is installed (e.g. Google Colab) and falls
back to the PyTorch backend otherwise.
"""
import importlib.util
import json
import os
import time
from collections import Counter

import numpy as np
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight

import config
from src import utils
from src.evaluation import cross_dataset_targets, record_result

log = utils.get_logger()


def _select_backend():
    if "KERAS_BACKEND" not in os.environ:
        for backend in ("tensorflow", "torch", "jax"):
            if importlib.util.find_spec(backend) is not None:
                os.environ["KERAS_BACKEND"] = backend
                break
    import keras
    return keras


# --------------------------------------------------------------------------
# Vocabulary / sequences
# --------------------------------------------------------------------------
PAD, OOV = 0, 1


def build_vocab(texts, max_vocab, min_freq):
    counts = Counter(w for t in texts for w in t.split())
    words = [w for w, c in counts.most_common(max_vocab - 2) if c >= min_freq]
    return {w: i + 2 for i, w in enumerate(words)}


def texts_to_sequences(texts, vocab, max_len):
    arr = np.zeros((len(texts), max_len), dtype=np.int32)
    for i, t in enumerate(texts):
        ids = [vocab.get(w, OOV) for w in t.split()][:max_len]
        arr[i, :len(ids)] = ids
    return arr


def build_model(keras, vocab_size, n_classes, p):
    from keras import layers
    inp = keras.Input(shape=(p["max_len"],), dtype="int32")
    x = layers.Embedding(vocab_size, p["embed_dim"], mask_zero=True)(inp)
    x = layers.Bidirectional(layers.LSTM(p["lstm_units"]))(x)
    x = layers.Dropout(p["dropout"])(x)
    x = layers.Dense(p["dense_units"], activation="relu")(x)
    x = layers.Dropout(p["dropout"])(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    model = keras.Model(inp, out)
    model.compile(optimizer=keras.optimizers.Adam(p["lr"]),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def _f1_checkpoint(keras, X_val, y_val, patience):
    class F1Checkpoint(keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.best_f1, self.best_weights, self.best_epoch, self.wait = -1.0, None, 0, 0
            self.history = []

        def on_epoch_end(self, epoch, logs=None):
            pred = self.model.predict(X_val, batch_size=512, verbose=0).argmax(axis=1)
            f1 = f1_score(y_val, pred, average="macro")
            self.history.append({"epoch": epoch + 1, "val_f1_macro": f1,
                                 "loss": (logs or {}).get("loss")})
            log.info(f"     epoch {epoch + 1}: val F1(macro)={f1:.4f}")
            if f1 > self.best_f1:
                self.best_f1, self.best_epoch, self.wait = f1, epoch + 1, 0
                self.best_weights = self.model.get_weights()
            else:
                self.wait += 1
                if self.wait >= patience:
                    self.model.stop_training = True

        def on_train_end(self, logs=None):
            if self.best_weights is not None:
                self.model.set_weights(self.best_weights)

    return F1Checkpoint()


def run_task(keras, dataset, task, df):
    p = config.BILSTM
    labels = utils.task_labels(task)
    tr, va, te = utils.split_frames(df)
    y_tr, y_va, y_te = (utils.encode_labels(x, task) for x in (tr, va, te))

    vocab = build_vocab(tr["text_proc"], p["max_vocab"], p["min_freq"])
    seq = lambda frame: texts_to_sequences(frame["text_proc"].tolist(), vocab, p["max_len"])
    X_tr, X_va, X_te = seq(tr), seq(va), seq(te)

    present = np.unique(y_tr)
    w = compute_class_weight("balanced", classes=present, y=y_tr)
    class_weight = {int(c): float(v) for c, v in zip(present, w)}

    keras.utils.set_random_seed(config.SEED)
    model = build_model(keras, len(vocab) + 2, len(labels), p)
    ckpt = _f1_checkpoint(keras, X_va, y_va, p["patience"])
    log.info(f"[bilstm] {dataset}/{task}: vocab={len(vocab) + 2:,} train={len(X_tr):,} "
             f"backend={keras.backend.backend()}")
    t0 = time.perf_counter()
    model.fit(X_tr, y_tr, validation_data=(X_va, y_va), epochs=p["epochs"],
              batch_size=p["batch_size"], class_weight=class_weight, callbacks=[ckpt], verbose=2)
    train_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    proba = model.predict(X_te, batch_size=512, verbose=0)
    latency_ms = 1000 * (time.perf_counter() - t0) / len(X_te)

    extra = {"imbalance_strategy": "class_weight", "train_time_s": train_time,
             "inference_ms_per_post": latency_ms, "n_train": int(len(y_tr)),
             "best_epoch": ckpt.best_epoch, "val_f1_macro": ckpt.best_f1,
             "epoch_history": ckpt.history, "vocab_size": len(vocab) + 2,
             "keras_backend": keras.backend.backend(), "hyperparameters": p}
    record_result(train_ds=dataset, eval_ds=dataset, task=task, model="bilstm",
                  variant="class_weight", is_main=True, y_true=y_te, proba=proba,
                  labels=labels, extra=extra)

    if task == "binary":
        for other in cross_dataset_targets(dataset):
            ote = utils.split_frames(utils.load_prepared(other))[2]
            pr = model.predict(seq(ote), batch_size=512, verbose=0)
            record_result(train_ds=dataset, eval_ds=other, task=task, model="bilstm",
                          variant="class_weight", is_main=True,
                          y_true=utils.encode_labels(ote, task), proba=pr, labels=labels,
                          extra=extra)
        mdir = utils.out_dir("models") / f"{dataset}__binary__bilstm"
        mdir.mkdir(parents=True, exist_ok=True)
        model.save(mdir / "model.keras")
        with open(mdir / "vocab.json", "w", encoding="utf-8") as f:
            json.dump({"vocab": vocab, "max_len": p["max_len"], "labels": labels}, f)


def run(datasets=None):
    keras = _select_backend()
    utils.set_seed()
    for ds in datasets or utils.available_prepared():
        df = utils.load_prepared(ds)
        for task in config.TASKS[ds]:
            run_task(keras, ds, task, df)
            keras.backend.clear_session()
