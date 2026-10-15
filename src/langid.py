"""Language identification worker (kept dependency-free so that parallel
worker processes import nothing else from the project)."""


def detect_batch(texts, min_words=3):
    from langdetect import DetectorFactory, detect
    DetectorFactory.seed = 0
    out = []
    for t in texts:
        if len(t.split()) < min_words:
            out.append("und")
            continue
        try:
            out.append(detect(t))
        except Exception:
            out.append("und")
    return out
