"""The GitHub Action's runtime: typed decisions on a CPU runner, no torch.

  python action/jev_action.py fetch --dir MODEL_DIR
  python action/jev_action.py run --dir MODEL_DIR --state-file s.json --workflow security_incidents
  python action/jev_action.py run --dir MODEL_DIR --state "..." --questions-file q.yml

Two paths, and every answer says which one produced it:

  ensemble - a built-in workflow's questions. Fine-tuned model blended with a
             frozen-encoder probe trained on that exact question, then
             sharpened. 0.697 on the typed-decisions test split.
  model    - your own questions. The fine-tuned model alone, schema supplied
             at request time. 0.62 on the same split.

A probe needs labelled examples for its question, which is why your own
questions cannot take the ensemble path. action/verify.py replays the full test
split through this file to produce both numbers.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import numpy as np

from typed_schema import TYPES, iter_labels, label_text, state_to_text

# Mirrors td_data (which imports torch). verify.py asserts the encodings match.
Q_TOKEN, L_TOKEN = "<<q>>", "<<l>>"
MAX_STATE_CHARS = 200_000     # only the first max_len tokens are read anyway
REPO = "intikhab49/open-jev-typed-decision-engine"


def read_sums():
    out = {}
    with open(os.path.join(HERE, "SHA256SUMS")) as f:
        for line in f:
            if line.strip():
                digest, name = line.split()
                out[name] = digest
    return out


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(model_dir, release, repo=REPO):
    """Download each release asset unless a verified copy is already there.

    Every file is checked against action/SHA256SUMS, which is committed in the
    same tag as this code - a swapped release asset fails here, not at decision
    time.
    """
    os.makedirs(model_dir, exist_ok=True)
    for name, digest in read_sums().items():
        path = os.path.join(model_dir, name)
        if os.path.exists(path) and sha256(path) == digest:
            print(f"  cached   {name}")
            continue
        url = f"https://github.com/{repo}/releases/download/{release}/{name}"
        print(f"  download {name}")
        part = path + ".part"
        urllib.request.urlretrieve(url, part)
        got = sha256(part)
        if got != digest:
            os.remove(part)
            sys.exit(f"{name}: sha256 {got} does not match SHA256SUMS ({digest})")
        os.replace(part, path)


def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def sharpen(p, temp):
    """Same as 07_ensemble.sharpen: p**(1/T), renormalised."""
    if temp == 1.0:
        return p
    q = np.power(np.clip(p, 1e-12, 1.0), 1.0 / temp)
    s = q.sum()
    return q / s if s > 0 else p


def parse_state(text):
    s = text.lstrip()
    if s[:1] in ("{", "["):
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            pass
    return text


def parse_questions(text):
    """JSON or YAML mapping of name -> {type, instructions, criteria}."""
    if not text.strip():
        return {}
    try:
        qs = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml
        except ImportError:
            sys.exit("questions are not valid JSON, and PyYAML is not installed to read YAML")
        qs = yaml.safe_load(text)
    if not isinstance(qs, dict):
        sys.exit("questions must be a mapping of name -> {type, instructions, criteria}")
    for name, q in qs.items():
        validate(str(name), q)
    return {str(k): v for k, v in qs.items()}


def validate(name, q):
    def bad(msg):
        sys.exit(f"question {name!r}: {msg}")
    if not isinstance(q, dict):
        bad("must be a mapping with type, instructions and (usually) criteria")
    if q.get("type") not in TYPES:
        bad(f"type must be one of {TYPES}, got {q.get('type')!r}")
    if not isinstance(q.get("instructions"), str) or not q["instructions"].strip():
        bad("instructions must be a non-empty string")
    crit = q.get("criteria")
    if q["type"] == "choice" and not (isinstance(crit, dict) and len(crit) >= 2):
        bad("a choice needs criteria: {label: description, ...} with at least two labels")
    if q["type"] == "score" and not (isinstance(crit, list) and len(crit) >= 2):
        bad("a score needs criteria: [level 0 description, level 1 description, ...]")
    if q["type"] == "noul" and crit is not None and set(crit) != {"true", "false"}:
        bad("a noul's criteria, if given, must have exactly the keys true and false")
    if crit is not None:
        labels = crit if isinstance(crit, dict) else {str(i): d for i, d in enumerate(crit)}
        for k, d in labels.items():
            if not isinstance(d, str):
                bad(f"criteria description for {k!r} must be a string")


class Engine:
    def __init__(self, model_dir):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.dir = model_dir
        self.cfg = json.load(open(os.path.join(model_dir, "config.json")))
        self.presets = json.load(open(os.path.join(model_dir, "presets.json")))
        self.tok = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
        self.ort = ort
        self.sess = ort.InferenceSession(os.path.join(model_dir, "jevlite.onnx"),
                                         providers=["CPUExecutionProvider"])
        self._enc = self._base_tok = self._probes = None
        ids = self.cfg["ids"]
        assert self.tok.token_to_id(Q_TOKEN) == ids["q"] and \
            self.tok.token_to_id(L_TOKEN) == ids["l"], "tokenizer/config mismatch"

    def _ids(self, text):
        return self.tok.encode(text, add_special_tokens=False).ids

    # --- fine-tuned model --------------------------------------------------
    def encode(self, state, questions):
        """td_data.encode without torch or gold. Questions are never truncated;
        the state absorbs the cut."""
        ids, max_len = self.cfg["ids"], self.cfg["max_len"]
        qnames = sorted(questions)
        tail, label_pos, spans = [], [], []
        for qname in qnames:
            q = questions[qname]
            tail += [ids["q"]] + self._ids(q["instructions"])
            labels = []
            for lab, desc in iter_labels(q):
                label_pos.append(len(tail))
                labels.append(lab)
                tail += [ids["l"]] + self._ids(label_text(lab, desc))
            spans.append((qname, labels))
        budget = max_len - len(tail) - 2
        if budget < 32:
            raise ValueError(
                f"these questions take {len(tail)} tokens, leaving no room for the "
                f"state in a {max_len}-token window - shorten or split them")
        text = state_to_text(state)[:MAX_STATE_CHARS]
        head = [ids["cls"]] + self._ids(text)[:budget]
        return head + tail + [ids["sep"]], [p + len(head) for p in label_pos], spans

    def model_dists(self, state, questions):
        """-> {qname: (labels, probs)} with per-type temperature applied, as
        model.predict_distributions does."""
        out, names = {}, sorted(questions)
        per = self.cfg["max_questions_per_pass"]
        for i in range(0, len(names), per):
            chunk = {n: questions[n] for n in names[i:i + per]}
            ids, pos, spans = self.encode(state, chunk)
            logits = self.sess.run(None, {
                "input_ids": np.array([ids], dtype=np.int64),
                "attention_mask": np.ones((1, len(ids)), dtype=np.int64),
                "label_pos": np.array([pos], dtype=np.int64)})[0][0]
            cursor = 0
            for qname, labels in spans:
                t = self.cfg["temperatures"][chunk[qname]["type"]]
                z = logits[cursor:cursor + len(labels)].astype("float64") / t
                out[qname] = (labels, softmax(z))
                cursor += len(labels)
        return out

    # --- frozen probe ------------------------------------------------------
    def embed(self, state):
        from tokenizers import Tokenizer
        if self._enc is None:
            self._enc = self.ort.InferenceSession(
                os.path.join(self.dir, "encoder.onnx"), providers=["CPUExecutionProvider"])
            self._base_tok = Tokenizer.from_file(os.path.join(self.dir, "tokenizer-base.json"))
            self._probes = json.load(open(os.path.join(self.dir, "probes.json")))
        ids = self.cfg["ids"]
        text = state_to_text(state)[:MAX_STATE_CHARS]
        body = self._base_tok.encode(text, add_special_tokens=False).ids
        seq = [ids["base_cls"]] + body[:self.cfg["probe_max_len"] - 2] + [ids["base_sep"]]
        return self._enc.run(None, {
            "input_ids": np.array([seq], dtype=np.int64),
            "attention_mask": np.ones((1, len(seq)), dtype=np.int64)})[0][0].astype("float64")

    def probe_dist(self, workflow, qname, labels, x):
        """probe.probe_distribution, from exported coefficients."""
        p, n = self._probes.get(f"{workflow}/{qname}"), len(labels)
        if p is None:
            return np.full(n, 1.0 / n)
        if "constant" in p:
            out = np.array([1.0 if l == p["constant"] else 0.0 for l in labels])
            return out if out.sum() else np.full(n, 1.0 / n)
        W, b = np.asarray(p["coef"]), np.asarray(p["intercept"])
        z = W @ x + b
        if len(p["classes"]) == 2 and W.shape[0] == 1:     # sklearn binary form
            p1 = 1.0 / (1.0 + np.exp(-z[0]))
            proba = np.array([1.0 - p1, p1])
        else:
            proba = softmax(z)
        by = dict(zip(p["classes"], proba))
        out = np.array([by.get(l, 0.0) for l in labels], dtype="float64")
        s = out.sum()
        return out / s if s > 0 else np.full(n, 1.0 / n)

    # --- public ------------------------------------------------------------
    def decide(self, state, workflow=None, questions=None):
        questions = questions or {}
        decisions = {}
        if workflow:
            if workflow not in self.presets:
                sys.exit(f"unknown workflow {workflow!r}; built-in: {sorted(self.presets)}")
            clash = sorted(set(questions) & set(self.presets[workflow]))
            if clash:
                sys.exit(f"custom question names clash with {workflow}'s built-ins: {clash}")
            preset = self.presets[workflow]
            md = self.model_dists(state, preset)      # all five together, as measured
            x = self.embed(state)
            w, temp = self.cfg["blend_w"], self.cfg["sharpen_t"]
            for qname, (labels, mp) in md.items():
                mix = w * mp + (1.0 - w) * self.probe_dist(workflow, qname, labels, x)
                s = mix.sum()
                mix = sharpen(mix / s if s > 0 else np.full(len(mix), 1.0 / len(mix)), temp)
                decisions[qname] = self._row(preset[qname], labels, mix, "ensemble")
        if questions:
            for qname, (labels, mp) in self.model_dists(state, questions).items():
                decisions[qname] = self._row(questions[qname], labels, mp, "model")
        return {"workflow": workflow or None, "decisions": decisions}

    @staticmethod
    def _row(q, labels, p, mode):
        k = int(np.argmax(p))
        return {"answer": labels[k], "confidence": round(float(p[k]), 4),
                "type": q["type"], "mode": mode,
                "distribution": {l: round(float(v), 4) for l, v in zip(labels, p)}}


def summary_md(result):
    head = f"### Open Jev decisions{' - ' + result['workflow'] if result['workflow'] else ''}\n\n"
    rows = ["| Question | Answer | Confidence | Path |", "|---|---|---|---|"]
    for qname, d in result["decisions"].items():
        rows.append(f"| `{qname}` | **{d['answer']}** | {d['confidence']:.2f} | {d['mode']} |")
    return head + "\n".join(rows) + "\n"


def run(a):
    if bool(a.state) == bool(a.state_file):
        sys.exit("give exactly one of state / state-file")
    text = open(a.state_file, encoding="utf-8").read() if a.state_file else a.state
    qtext = open(a.questions_file, encoding="utf-8").read() if a.questions_file \
        else (a.questions or "")
    questions = parse_questions(qtext)
    if not a.workflow and not questions:
        sys.exit("nothing to decide: give a built-in workflow, questions, or both")

    result = Engine(a.dir).decide(parse_state(text), a.workflow or None, questions)
    blob = json.dumps(result, ensure_ascii=False)
    print(json.dumps(result, indent=1, ensure_ascii=False))
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(blob)
    gh_out = os.environ.get("GITHUB_OUTPUT")
    if gh_out:
        with open(gh_out, "a", encoding="utf-8") as f:
            f.write(f"result<<OPEN_JEV_EOF\n{blob}\nOPEN_JEV_EOF\n")
            if a.out:
                f.write(f"result-file={a.out}\n")
    gh_sum = os.environ.get("GITHUB_STEP_SUMMARY")
    if gh_sum:
        with open(gh_sum, "a", encoding="utf-8") as f:
            f.write(summary_md(result))


if __name__ == "__main__":
    env = os.environ.get
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--dir", default=env("JEV_MODEL_DIR", "jev-model"))
    f.add_argument("--release", default=env("JEV_MODEL_RELEASE", "model-v1"))
    f.add_argument("--repo", default=REPO)
    r = sub.add_parser("run")
    r.add_argument("--dir", default=env("JEV_MODEL_DIR", "jev-model"))
    r.add_argument("--state", default=env("JEV_STATE", ""))
    r.add_argument("--state-file", default=env("JEV_STATE_FILE", ""))
    r.add_argument("--workflow", default=env("JEV_WORKFLOW", ""))
    r.add_argument("--questions", default=env("JEV_QUESTIONS", ""))
    r.add_argument("--questions-file", default=env("JEV_QUESTIONS_FILE", ""))
    r.add_argument("--out", default=env("JEV_RESULT_FILE", ""))
    a = ap.parse_args()
    if a.cmd == "fetch":
        fetch(a.dir, a.release, a.repo)
    else:
        run(a)
