"""Step 3 - fit one temperature per primitive type on the validation split.

This is the cheapest win available and it targets Jev's weakest measured claim.
Independent testing found Jev's miscalibration runs in *opposite directions* by
type: choice and score overconfident (refit T around 3.3), noul underconfident
(T around 0.66). A single global temperature cannot repair both at once, so we
fit three and store them in the checkpoint.

Runs on CPU in seconds. Nothing here touches the encoder weights.
"""
import argparse
import torch
from torch.utils.data import DataLoader
from td_data import load_split, TypedDecisions, collate, val_split, pad_cat
from model import JevLite, grouped_softmax, argmax_per_question, require_ckpt
from typed_schema import TYPES
from metrics import ece, ece_confidence, brier, nll


def collect(model, dl, dev):
    """Cache raw logits once so the temperature search is a pure CPU fit."""
    out = []
    model.eval()
    with torch.no_grad():
        for b in dl:
            b = {k: v.to(dev) for k, v in b.items()}
            lg = model.logits(b["input_ids"], b["attention_mask"], b["label_pos"])
            out.append({k: v.cpu() for k, v in b.items()} | {"logits": lg.float().cpu()})
    return out


def apply_temp(batch, log_temp):
    g = batch["group"].clamp(min=0)
    t = batch["qtype"].clamp(min=0).gather(1, g)
    lg = batch["logits"] / log_temp.exp()[t]
    return grouped_softmax(lg, batch["group"], batch["label_mask"],
                           batch["qtype"].size(1))


T_MIN, T_MAX = 0.25, 5.0
MIN_DECISIONS = 30


def _bounded(raw):
    """T in [T_MIN, T_MAX] via sigmoid, so the search cannot run to infinity.

    Unbounded NLL minimisation against *soft* targets has a degenerate optimum:
    when the consensus is near-uniform, T -> inf flattens every prediction to
    uniform and scores well on NLL while destroying the confidence signal the
    whole point of this step was to make trustworthy. Ask for T=400 and you get
    a model that is beautifully calibrated and useless for routing.
    """
    return T_MIN + (T_MAX - T_MIN) * torch.sigmoid(raw)


def type_counts(cached, n_types=len(TYPES)):
    counts = torch.zeros(n_types, dtype=torch.long)
    for b in cached:
        ty = b["qtype"][b["qmask"]]
        for i in range(n_types):
            counts[i] += int((ty == i).sum())
    return counts


def fit(cached, n_types=len(TYPES), iters=200):
    # sigmoid(0) = 0.5 -> T starts at the midpoint of the bounded range; offset
    # so the initial value is 1.0 (a no-op temperature).
    init = torch.log(torch.tensor((1.0 - T_MIN) / (T_MAX - 1.0)))
    raw = torch.full((n_types,), float(init), requires_grad=True)
    opt = torch.optim.LBFGS([raw], lr=0.1, max_iter=iters)

    def closure():
        opt.zero_grad()
        loss = 0.0
        for b in cached:
            p = apply_temp(b, _bounded(raw).log()).clamp_min(1e-8)
            m = b["label_mask"].float()
            loss = loss - (b["target"] * p.log() * m).sum() / b["qmask"].sum()
        loss.backward()
        return loss

    opt.step(closure)
    temps = _bounded(raw).detach()

    # A type with too few validation decisions gets T=1 rather than a fit to noise.
    counts = type_counts(cached, n_types)
    for i, c in enumerate(counts):
        if c < MIN_DECISIONS:
            print(f"  ! {TYPES[i]}: only {int(c)} validation decisions "
                  f"(< {MIN_DECISIONS}) - leaving T=1.0 instead of fitting noise")
            temps[i] = 1.0
    at_bound = [(TYPES[i], float(t)) for i, t in enumerate(temps)
                if t <= T_MIN * 1.01 or t >= T_MAX * 0.99]
    for name, t in at_bound:
        print(f"  ! {name}: T={t:.2f} hit the bound - the soft targets for this "
              "type are close to uniform; treat its confidence with suspicion")
    return temps.log()


def report(cached, log_temp, tag):
    ps, ts, ms, tys, confs, corrects, qts = [], [], [], [], [], [], []
    for b in cached:
        p = apply_temp(b, log_temp)
        pred, conf = argmax_per_question(p, b)
        gold, _ = argmax_per_question(b["target"], b)
        m = b["qmask"]
        confs.append(conf[m]); corrects.append((pred == gold)[m])
        qts.append(b["qtype"][m])
        ps.append(p); ts.append(b["target"]); ms.append(b["label_mask"])
        tys.append(b["qtype"].clamp(min=0).gather(1, b["group"].clamp(min=0)))
    P, T = pad_cat(ps), pad_cat(ts)
    M, TY = pad_cat(ms, False), pad_cat(tys)
    C, OK, QT = torch.cat(confs), torch.cat(corrects), torch.cat(qts)
    overall = ece_confidence(C, OK)
    print(f"  {tag:<12} ECE {overall:.4f}   "
          f"ECE-dist {ece(P, T, M):.4f}   Brier {brier(P, T, M):.4f}   "
          f"NLL {nll(P, T, M):.4f}")
    for i, name in enumerate(TYPES):
        sel = QT == i
        if sel.any():
            print(f"      {name:<8} ECE {ece_confidence(C[sel], OK[sel]):.4f}"
                  f"   mean conf {C[sel].mean():.3f}   acc {OK[sel].float().mean():.3f}")
    return overall


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="jevlite.pt")
    p.add_argument("--bs", type=int, default=8)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--config", default="all",
                   help="one workflow (customer_service, invoice_processing, "
                        "security_incidents, agent_trace_observability) instead "
                        "of all four - a domain specialist is where a small model wins")
    a = p.parse_args()

    ck = require_ckpt(a.ckpt)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = JevLite(ck["encoder"])
    model.load_state_dict(ck["state_dict"])
    model.to(dev).eval()

    _, va = val_split(load_split("train", a.config, limit=a.limit))
    ds = TypedDecisions(va, model.tok, ck["max_len"], model.qid, model.lid)
    dl = DataLoader(ds, batch_size=a.bs,
                    collate_fn=lambda b: collate(b, model.tok.pad_token_id))
    cached = collect(model, dl, dev)

    print(f"fitting on {len(va)} validation cases\n")
    ece_before = report(cached, torch.zeros(len(TYPES)), "before")
    log_temp = fit(cached)
    print()
    ece_after = report(cached, log_temp, "after")

    # Measured on the first full run: training against the soft consensus
    # distribution already calibrates the model, the fitted temperatures came
    # back at ~1.0, and applying them made ECE slightly WORSE. So this step is
    # now a selection, not an assumption - it keeps T=1 unless it earns its place.
    if ece_after >= ece_before:
        print(f"\n  temperature scaling did not help "
              f"({ece_before:.4f} -> {ece_after:.4f}) - keeping T=1.0.")
        print("  That is the expected outcome when the loss already trains on the")
        print("  consensus distribution; it means your calibration came for free.")
        log_temp = torch.zeros(len(TYPES))
    else:
        print(f"\n  temperature scaling helped: ECE {ece_before:.4f} -> "
              f"{ece_after:.4f}")

    print("\n  fitted temperatures:  " + "  ".join(
        f"{n}={log_temp.exp()[i]:.3f}" for i, n in enumerate(TYPES)))
    print("  (T > 1 means the model was overconfident on that type, T < 1 under)")

    model.log_temp.data = log_temp
    ck["state_dict"] = model.state_dict()
    ck["temperatures"] = log_temp.exp().tolist()
    torch.save(ck, a.ckpt)
    print(f"\nsaved -> {a.ckpt}")
