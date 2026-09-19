"""Step 3 - the baseline your neural model has to beat.

TF-IDF + logistic regression, plus a majority-class floor and a regex rule for
is_terminal_error. Runs on CPU in seconds. If the 150M-param encoder in step 4
does not clearly beat these numbers, the encoder is not earning its keep.
"""
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from data import load, split, targets
from schema import ACTIONS

ERR_RE = re.compile(
    r"traceback|\berror\b|exception|assertionerror|segmentation fault|"
    r"fatal|compile(r)? error|exit code [1-9]|FAILED|panic:",
    re.I,
)

if __name__ == "__main__":
    s = split(load())
    tr, te = s["train"], s["test"]
    print(f"train {len(tr)}  val {len(s['val'])}  test {len(te)}\n")

    Xtr = [r["text"] for r in tr]
    Xte = [r["text"] for r in te]
    ytr = targets(tr)
    yte = targets(te)

    vec = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), sublinear_tf=True)
    Vtr, Vte = vec.fit_transform(Xtr), vec.transform(Xte)

    names = ["is_terminal_error", "action_required", "confidence_score"]
    print(f"{'field':<20}{'majority':>10}{'regex':>9}{'tfidf+lr':>11}")
    print("-" * 50)
    for k, name in enumerate(names):
        maj = max(np.bincount(ytr[k].astype(int))) and \
              accuracy_score(yte[k].astype(int),
                             np.full(len(yte[k]), np.bincount(ytr[k].astype(int)).argmax()))
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        clf.fit(Vtr, ytr[k].astype(int))
        lr = accuracy_score(yte[k].astype(int), clf.predict(Vte))
        rx = ""
        if name == "is_terminal_error":
            pred = np.array([bool(ERR_RE.search(t)) for t in Xte]).astype(int)
            rx = f"{accuracy_score(yte[k].astype(int), pred):9.3f}"
        print(f"{name:<20}{maj:10.3f}{rx if rx else '        -':>9}{lr:11.3f}")

    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(Vtr, ytr[1])
    print(f"\naction_required macro-F1 (tfidf+lr): "
          f"{f1_score(yte[1], clf.predict(Vte), average='macro'):.3f}")
    print(f"classes: {ACTIONS}")
