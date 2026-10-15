"""Models 5-6: BERT (bert-base-uncased) and AfriBERTa (castorini/afriberta_base),
fine-tuned with HuggingFace Transformers + PyTorch.

* Input: `text_clean` (cleaned natural text; stop-words are NOT removed because
  transformers rely on full sentence context). Each model uses its own
  sub-word tokeniser (WordPiece for BERT, SentencePiece for AfriBERTa).
* Class imbalance: class-weighted cross-entropy (SMOTE cannot be applied to
  raw text sequences).
* The epoch with the best validation macro F1 is kept.
* A GPU is strongly recommended (Google Colab / Kaggle notebooks, as stated in
  the proposal's limitations section). Mixed precision is used on CUDA.
"""
import math
import time

import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight

import config
from src import utils
from src.evaluation import cross_dataset_targets, record_result

log = utils.get_logger()


def _device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(hf_name, n_labels):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(hf_name)
    model = AutoModelForSequenceClassification.from_pretrained(hf_name, num_labels=n_labels)
    return tok, model


def _encode(tok, texts, device):
    enc = tok(list(texts), padding=True, truncation=True, max_length=config.TRANSFORMER["max_len"],
              return_tensors="pt")
    return {k: v.to(device) for k, v in enc.items()}


@torch.no_grad()
def predict_proba(model, tok, texts, device, batch_size=None):
    batch_size = batch_size or config.TRANSFORMER["eval_batch_size"]
    model.eval()
    texts = list(texts)
    # sort by length for efficient padding, then restore order
    order = np.argsort([len(t) for t in texts])
    out = np.zeros((len(texts), model.config.num_labels), dtype=np.float32)
    use_amp = device.type == "cuda"
    for i in range(0, len(texts), batch_size):
        idx = order[i:i + batch_size]
        enc = _encode(tok, [texts[j] for j in idx], device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(**enc).logits
        out[idx] = torch.softmax(logits.float(), dim=-1).cpu().numpy()
    return out


def fine_tune(model, tok, train_texts, y_train, val_texts, y_val, n_labels, device):
    p = config.TRANSFORMER
    model.to(device)
    present = np.unique(y_train)
    w = np.ones(n_labels, dtype=np.float32)
    w[present] = compute_class_weight("balanced", classes=present, y=y_train)
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(w, device=device))

    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    groups = [
        {"params": [p_ for n, p_ in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": p["weight_decay"]},
        {"params": [p_ for n, p_ in model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(groups, lr=p["lr"])
    steps_per_epoch = math.ceil(len(train_texts) / p["batch_size"])
    total = steps_per_epoch * p["epochs"]
    from transformers import get_linear_schedule_with_warmup
    sched = get_linear_schedule_with_warmup(opt, int(p["warmup_ratio"] * total), total)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    train_texts = np.asarray(list(train_texts), dtype=object)
    rng = np.random.default_rng(config.SEED)
    best_f1, best_state, best_epoch, history = -1.0, None, 0, []
    for epoch in range(1, p["epochs"] + 1):
        model.train()
        perm = rng.permutation(len(train_texts))
        running = 0.0
        for step in range(steps_per_epoch):
            idx = perm[step * p["batch_size"]:(step + 1) * p["batch_size"]]
            enc = _encode(tok, train_texts[idx], device)
            yb = torch.tensor(y_train[idx], device=device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(**enc).logits
            loss = loss_fn(logits.float(), yb)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), p["max_grad_norm"])
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            sched.step()
            running += loss.item()
            if (step + 1) % p["log_every"] == 0:
                log.info(f"     epoch {epoch} step {step + 1}/{steps_per_epoch} "
                         f"loss={running / (step + 1):.4f}")
        val_pred = predict_proba(model, tok, val_texts, device).argmax(axis=1)
        f1 = f1_score(y_val, val_pred, average="macro")
        history.append({"epoch": epoch, "train_loss": running / steps_per_epoch, "val_f1_macro": f1})
        log.info(f"     epoch {epoch}: train loss={running / steps_per_epoch:.4f} val F1(macro)={f1:.4f}")
        if f1 > best_f1:
            best_f1, best_epoch = f1, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return {"val_f1_macro": best_f1, "best_epoch": best_epoch, "epoch_history": history}


def run_task(model_key, hf_name, dataset, task, df, device):
    labels = utils.task_labels(task)
    tr, va, te = utils.split_frames(df)
    y_tr, y_va, y_te = (utils.encode_labels(x, task) for x in (tr, va, te))
    utils.set_seed()
    log.info(f"[transformer] {model_key} ({hf_name}) {dataset}/{task} on {device}: "
             f"train={len(tr):,} val={len(va):,} test={len(te):,}")
    tok, model = load_model(hf_name, len(labels))

    t0 = time.perf_counter()
    info = fine_tune(model, tok, tr["text_clean"], y_tr, va["text_clean"], y_va, len(labels), device)
    train_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    proba = predict_proba(model, tok, te["text_clean"], device)
    latency_ms = 1000 * (time.perf_counter() - t0) / len(te)

    extra = {"imbalance_strategy": "class_weight", "hf_model": hf_name, "device": str(device),
             "train_time_s": train_time, "inference_ms_per_post": latency_ms,
             "n_train": int(len(y_tr)), "hyperparameters": config.TRANSFORMER,
             "n_parameters": int(sum(p.numel() for p in model.parameters())), **info}
    record_result(train_ds=dataset, eval_ds=dataset, task=task, model=model_key,
                  variant="class_weight", is_main=True, y_true=y_te, proba=proba,
                  labels=labels, extra=extra)

    if task == "binary":
        for other in cross_dataset_targets(dataset):
            ote = utils.split_frames(utils.load_prepared(other))[2]
            pr = predict_proba(model, tok, ote["text_clean"], device)
            record_result(train_ds=dataset, eval_ds=other, task=task, model=model_key,
                          variant="class_weight", is_main=True,
                          y_true=utils.encode_labels(ote, task), proba=pr, labels=labels,
                          extra=extra)
        if config.SAVE_TRANSFORMER_BINARY_MODELS:
            mdir = utils.out_dir("models") / f"{dataset}__binary__{model_key}"
            model.save_pretrained(mdir)
            tok.save_pretrained(mdir)

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()


def run(datasets=None, models=None):
    device = _device()
    if device.type == "cpu":
        log.warning("[transformer] No GPU detected - fine-tuning BERT/AfriBERTa on CPU will be "
                    "very slow. Use Google Colab/Kaggle (see README) or --max-rows.")
    models = models or list(config.TRANSFORMER_MODELS)
    for ds in datasets or utils.available_prepared():
        df = utils.load_prepared(ds)
        for task in config.TASKS[ds]:
            for key in models:
                run_task(key, config.TRANSFORMER_MODELS[key], ds, task, df, device)
