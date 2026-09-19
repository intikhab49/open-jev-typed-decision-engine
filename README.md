<div align="center">

# Open Jev — a typed decision engine you can train for free

**A 150M encoder that answers arbitrary typed questions about a state in one forward pass, with calibrated confidence. 0.03 behind TypeSafe Jev on its own benchmark, 2.5× better calibrated, 4× faster, $0.**

[![License](https://img.shields.io/badge/license-Apache%202.0-2a78d6?style=for-the-badge)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-2a78d6?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Colab](https://img.shields.io/badge/train%20on-free%20T4-eb6834?style=for-the-badge&logo=googlecolab&logoColor=white)](jevlite_colab.ipynb)
[![Benchmark](https://img.shields.io/badge/benchmark-typed--decisions-1baf7a?style=for-the-badge)](https://huggingface.co/datasets/LocalLLaMA/typed-decisions)

[Results](#results) · [How it works](#how-it-works) · [Quick start](#quick-start) · [What we learned](#what-the-runs-actually-proved) · [Limitations](#limitations)

</div>

---

## What this is

[TypeSafe AI's **Jev**](https://www.mindstudio.ai/blog/jev-system-one-model-launch) (launched 2026-09-15) is a "System One" model: it never writes prose, it only **decides, classifies, routes and scores**. You hand it a state plus a set of typed questions and it answers all of them in one non-autoregressive pass, with a confidence on each. $0.042/1M input, output tokens free, ~239 ms per call.

This repo reproduces that interface with an open 150M encoder you can train on a free Colab T4 in under 30 minutes, then run locally — or in a browser — for nothing.

Three primitives, matching theirs:

| type | meaning | criteria format |
|---|---|---|
| `noul` | boolean | `{"true": desc, "false": desc}`, or omitted entirely |
| `choice` | enum | `{label: desc, ...}` |
| `score` | ordered scale | `[level_0, level_1, ...]` |

```python
decide(state, {
    "action": Question.choice("What should the observability system do?", {
        "continue": "Let the agent proceed.",
        "human_review": "Queue this trace for a human.",
        "stop": "Halt the agent now."}),
    "risk": Question.score("How risky was this behaviour?", [
        "Benign: read-only.", "Low: routine writes.",
        "Moderate: irreversible.", "High: destructive."]),
})
# -> {"action": {"label": "human_review", "confidence": 0.54, "probabilities": {...}},
#     "risk":   {"label": "2",            "confidence": 0.47, "probabilities": {...}}}
```

The questions are supplied **per request**. Nothing about the schema is baked into the weights.

---

## Results

Measured on the official 400-case test split of [`LocalLLaMA/typed-decisions`](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) (2,000 decisions across four workflows). Blend weight and temperature were chosen on validation, never on test.

| configuration | accuracy | ECE ↓ | schema |
|---|---|---|---|
| Fine-tuned model alone | 0.6240 | 0.1045 | **dynamic** |
| Frozen probe alone | 0.6705 | **0.0389** | fixed |
| **Ensemble** (w=0.60, T=0.35) | **0.6965** | 0.0565 | fixed |
| TypeSafe Jev 1.13.0 | **0.7270** | 0.1440 | dynamic |
| *majority-class baseline* | *0.4830* | — | — |
| *single human annotator* | *0.6590* | — | — |

**0.03 behind Jev on accuracy. 2.5× better calibrated. ~60 ms per case locally against their 239 ms p50. Free, offline, open weights.**

### Can you route on the confidence?

This is the question that decides whether a decision model is usable. Ensemble, on test:

| confidence ≥ | coverage | accuracy |
|---|---|---|
| 0.90 | 25.4% | **0.915** |
| 0.80 | 44.5% | 0.846 |
| 0.70 | 61.7% | 0.806 |
| 0.50 | 91.9% | 0.726 |

A quarter of all decisions at 91.5% accuracy is a shippable policy: autoroute that band, escalate the rest.

---

## How it works

### The label-embedding head

The obvious design is one `nn.Linear` per question. It works, and it freezes the schema into the weights — add a question or rename a label and you retrain.

Instead, **every candidate label is written into the input sequence** behind a `<<l>>` marker token, and the head is a single `Linear(d, 1)` read at each marker. Softmax runs within each question's own label set.

```
[CLS] {state json} <<q>> How risky was this? <<l>> 0: Benign… <<l>> 1: Low… <<q>> …
                   ↑ question marker         ↑ one logit read off each of these
```

- The schema is **data at inference time** — new questions, new labels, a whole new workflow, with no retraining and no new parameters.
- Labels attend to the state *and to each other*: a cross-encoder, not a bi-encoder.
- One encoder pass answers every question about a state.

Base encoder is **ModernBERT-base** (150M, bidirectional, 8192 context). A causal model works via `--encoder`, but pools differently — token 0 in a decoder-only model has only seen itself, so `[CLS]` pooling on one is a bug, not a shortcut.

### Training on distributions, not labels

The dataset ships **full annotator probability distributions**, not just argmax labels. Training on the argmax throws that away and directly causes overconfidence: a model taught that a 55/45 case is `true` learns to say 0.95, then is wrong 45% of the time while claiming near-certainty.

So the loss is soft cross-entropy against the consensus distribution plus a Brier term — a proper scoring rule, which is what actually pushes probabilities toward honesty.

---

## Quick start

### Colab (free T4, ~30 min, no API key)

Upload [`jevlite_colab.ipynb`](jevlite_colab.ipynb) — the whole repo is embedded in it — set **Runtime → T4 GPU**, and **Run all**.

### Local

```bash
pip install -r requirements.txt

python smoke_test.py --all      # validate all 1,600 rows, no GPU, ~2 min
python 01_ceiling.py            # baselines you have to beat, $0
python 02_train.py --epochs 20  # ~25 min on a T4
python 03_calibrate.py          # per-type temperature, CPU, seconds
python 04_eval.py               # official test split, vs Jev
python 07_ensemble.py           # blend with the frozen probe
python plots.py --outdir plots  # six charts
python 05_serve.py --serve      # POST /decide {"state": ..., "questions": ...}
python 06_export_onnx.py --quantize
```

Every script takes `--limit N` for a fast dry run, and `--config <workflow>` to train a single-domain specialist.

### Files

| | |
|---|---|
| `typed_schema.py` | the three primitives; `Question.noul/choice/score` builders |
| `td_data.py` | dataset loading (HTTP fallback), marker-token encoding, collate |
| `model.py` | `JevLite`, grouped softmax, soft-CE + Brier loss |
| `probe.py` | frozen-encoder + logistic-regression baseline |
| `metrics.py` | ECE (two kinds), Brier, NLL, TVD, risk-coverage |
| `plots.py` | six charts, rendered from the JSON each step writes |
| `01_…` → `07_…` | the pipeline |
| `smoke_test.py` | correctness checks, no GPU needed |
| `legacy/` | the earlier fixed-head version, kept for comparison |

---

## What the runs actually proved

**A frozen encoder + logistic regression scores 0.670 — beating the fine-tuned model's 0.6240.** No training at all. This reproduces the [independent Banking77 finding](https://github.com/ickma2311/jev-baselines-eval) where the same baseline beat Jev by 10 points at 44× the speed. `01_ceiling.py` runs it first so you know what you're up against. If your fine-tune can't clear it, ship the probe.

**The accuracy gap was undertraining — up to a point.** 6 epochs gave 0.568; 20 epochs with early stopping (best at epoch 9) gave 0.6195. Past that, training loss kept falling while validation flatlined. The binding constraint is **1,016 training cases, not model size or epochs** — which is why reaching for a bigger encoder first is the wrong instinct here.

**Post-hoc temperature scaling had three different outcomes, so don't generalise from one.** A no-op on the undertrained model (temperatures ≈ 1.0, ECE slightly worse), a mild help on the overfit one, and decisive on the ensemble: **ECE 0.156 → 0.057 with accuracy untouched.** Mixing two disagreeing distributions flattens them, so the blend was badly *under*confident; sharpening in probability space can't move the argmax, so it's free or nothing. Before sharpening, the ≥0.90 band held 0.5% of traffic; after, 25.4%.

**Single-annotator agreement is not a ceiling.** It's 0.659 here — what one human rater scores against the consensus. A model that always picks the consensus argmax scores 1.000, and Jev already exceeds it. The real limit is that **15.6% of test decisions are near-ties** (top two labels within 0.1), so the last few points of headline accuracy are mostly luck. Judge on calibration and coverage.

---

## Limitations

**The ensemble trades away the dynamic schema.** The probe only answers questions it has training labels for. On an unseen question it returns uniform and the blend degrades to 0.6 × the neural model — so **0.6965 is the known-schema number and ~0.624 is the new-schema number.** Three configurations, different trade-offs, no single best. Always say which one a number came from.

**Jev's 0.727 is zero-shot.** It never saw this dataset; this model trained on 1,016 in-domain cases from the same distribution and still came in lower. That makes the gap look worse, not better, and it's the accurate framing.

**Small evaluation.** 400 test cases / 2,000 decisions. Differences under ~0.02 are not meaningful.

**Jev's figures are quoted, not reproduced here.** Accuracy 0.727 and ECE 0.144 come from the typed-decisions benchmark; latency and pricing from the independent evaluations linked below.

---

## Traps worth knowing if you fork this

- **10% of the dataset is a question shape the first rows never show.** Bare `noul` questions carry no `criteria` key at all. A 6-row smoke test passes and training then dies 300 rows in. `smoke_test.py --all` encodes every row for exactly this reason.
- **Workflows disagree on label width** (16/17/18/20), so `torch.cat` across batches fails the moment a run spans more than one. Hence `pad_cat`.
- **Unbounded temperature scaling on soft targets is degenerate** — the first fit returned T=400, which flattens everything to uniform, minimises NLL beautifully, and destroys the confidence signal. Bounded to [0.25, 5.0], and refused for any type with under 30 validation decisions.
- **ONNX export below opset 18 produces an invalid graph** — torch emits `Split` with `num_outputs` regardless, and ORT then refuses to load it. Silent at export, fatal at load; hence the parity check.
- **ONNX weights land in a 596 MB `.onnx.data` sidecar.** Serve both files or you load a graph with no weights. int8 (151 MB) shifts logits materially — recalibrate rather than reusing fp32 temperatures.
- **Never lower `--max-len`.** States run 401–997 tokens (p50 588); 512 truncates 73% of them. Batches already pad to their own longest example, so the high cap costs nothing.

---

## Credits

- **TypeSafe AI** for [Jev](https://www.mindstudio.ai/blog/jev-system-one-model-launch), the design this reproduces. Not affiliated with, endorsed by, or derived from their model.
- [`LocalLLaMA/typed-decisions`](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) — the public benchmark.
- [`answerdotai/ModernBERT-base`](https://huggingface.co/answerdotai/ModernBERT-base) — the encoder.
- Prior open reproductions: [Verdict / OpenJev](https://github.com/Heman10x-NGU/Verdict-open-jev) (ModernBERT-151M + WebGPU playground) and [open-jev-deberta-v3-large](https://huggingface.co/com-kotobalabs/open-jev-deberta-v3-large).
- Independent evaluations this repo's claims lean on: [baselines eval](https://github.com/ickma2311/jev-baselines-eval) (Banking77 / CLINC150), [phishing bench](https://github.com/anisselbd/jev-phishing-bench) (vs Claude Haiku 4.5), [OOD calibration](https://github.com/scienthoon/jev-ood-calibration) (per-type ECE).

## License

[Apache-2.0](LICENSE).

---

<div align="center">

**Keywords:** typed decision model · System One model · non-autoregressive classifier · TypeSafe Jev alternative · open source Jev · ModernBERT · structured output · calibrated confidence · LLM routing · agent observability · confidence thresholding · expected calibration error

</div>
