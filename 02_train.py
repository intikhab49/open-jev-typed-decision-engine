"""Step 2 - fine-tune JevLite on typed-decisions. Colab T4, ~15 min.

  python 02_train.py --epochs 6

Trains against the consensus *distribution* (soft CE + Brier), not the argmax.
That is the whole calibration story: a model taught that a 55/45 case is 55/45
reports 0.55, while a model taught it is "true" reports 0.95 and is then wrong
45% of the time while claiming near-certainty.
"""
import argparse
import json
import time
import torch
from torch.utils.data import DataLoader
from transformers import (get_cosine_schedule_with_warmup,
                          get_linear_schedule_with_warmup)
from td_data import load_split, TypedDecisions, collate, val_split
from model import JevLite, decision_loss, argmax_per_question, DEFAULT_ENCODER



def accuracy(model, dl, dev, temp=False):
    model.eval()
    hit = tot = 0
    with torch.no_grad():
        for b in dl:
            b = {k: v.to(dev) for k, v in b.items()}
            _, probs = model(b, apply_temperature=temp)
            pred, _ = argmax_per_question(probs, b)
            gold, _ = argmax_per_question(b["target"], b)
            m = b["qmask"]
            hit += ((pred == gold) & m).sum().item()
            tot += m.sum().item()
    return hit / max(tot, 1)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--encoder", default=DEFAULT_ENCODER)
    p.add_argument("--epochs", type=int, default=20,
                   help="6 was measurably too few - validation accuracy was still "
                        "climbing steeply when the first run stopped")
    p.add_argument("--patience", type=int, default=6,
                   help="stop after this many epochs with no validation gain; "
                        "0 disables. The best checkpoint is kept either way")
    p.add_argument("--schedule", default="cosine", choices=["cosine", "linear"])
    p.add_argument("--bs", type=int, default=4)
    p.add_argument("--accum", type=int, default=4)
    p.add_argument("--max-len", type=int, default=1024)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--head-lr", type=float, default=1e-3)
    p.add_argument("--brier", type=float, default=1.0)
    p.add_argument("--out", default="jevlite.pt")
    p.add_argument("--config", default="all",
                   help="one workflow (customer_service, invoice_processing, "
                        "security_incidents, agent_trace_observability) instead "
                        "of all four - a domain specialist is where a small model wins")
    p.add_argument("--limit", type=int, default=None,
                   help="tiny run to verify the loop before spending GPU time")
    a = p.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = JevLite(a.encoder).to(dev)
    tok = model.tok
    tr_rows, va_rows = val_split(load_split("train", a.config, limit=a.limit))

    def mk(rows):
        return TypedDecisions(rows, tok, a.max_len, model.qid, model.lid)

    def coll(b):
        return collate(b, tok.pad_token_id)

    tl = DataLoader(mk(tr_rows), batch_size=a.bs, shuffle=True, collate_fn=coll)
    vl = DataLoader(mk(va_rows), batch_size=a.bs, collate_fn=coll)
    print(f"device {dev} | train {len(tr_rows)} | val {len(va_rows)} "
          f"| effective batch {a.bs * a.accum}")

    head = [q for n, q in model.named_parameters() if n.startswith("score.")]
    enc = [q for n, q in model.named_parameters()
           if not n.startswith("score.") and q.requires_grad]
    opt = torch.optim.AdamW([{"params": enc, "lr": a.lr},
                             {"params": head, "lr": a.head_lr}], weight_decay=0.01)
    steps = max(1, (len(tl) // a.accum) * a.epochs)
    mk_sched = (get_cosine_schedule_with_warmup if a.schedule == "cosine"
                else get_linear_schedule_with_warmup)
    sched = mk_sched(opt, int(0.1 * steps), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=(dev == "cuda"))

    best, history, stale = -1.0, [], 0
    # An epoch is minutes long and silence is indistinguishable from a hang, so
    # emit a heartbeat. Colab in particular gives no other signal that a cell is
    # alive rather than wedged.
    every = max(1, len(tl) // 10)
    for ep in range(a.epochs):
        model.train()
        run = 0.0
        t_ep = time.time()
        for i, b in enumerate(tl):
            b = {k: v.to(dev) for k, v in b.items()}
            with torch.autocast("cuda", dtype=torch.float16, enabled=(dev == "cuda")):
                _, probs = model(b)
            loss, ce, br = decision_loss(probs, b, a.brier)
            scaler.scale(loss / a.accum).backward()
            run += loss.item()
            if (i + 1) % every == 0:
                done = (i + 1) / len(tl)
                el = time.time() - t_ep
                print(f"  epoch {ep+1}  {done:4.0%}  loss {run/(i+1):.4f}  "
                      f"{el:.0f}s elapsed, ~{el/done - el:.0f}s left", flush=True)
            if (i + 1) % a.accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                opt.zero_grad(set_to_none=True)
        acc = accuracy(model, vl, dev)
        star = "  *" if acc > best else ""
        print(f"epoch {ep+1}/{a.epochs}  loss {run/len(tl):.4f}  val acc {acc:.4f}{star}")
        # Written every epoch, not at the end: a reclaimed Colab session still
        # leaves you the curve that tells you whether to train longer.
        history.append({"epoch": ep + 1, "loss": run / len(tl), "val_acc": acc})
        json.dump(history, open("history.json", "w"), indent=1)

        if acc > best:
            stale, best = 0, acc
            torch.save({"state_dict": model.state_dict(), "encoder": a.encoder,
                        "max_len": a.max_len, "val_acc": acc}, a.out)
        else:
            stale += 1
            if a.patience and stale >= a.patience:
                print(f"\nno gain for {stale} epochs - stopping at epoch {ep+1}. "
                      f"The best checkpoint ({best:.4f}) is what was kept.")
                break
    print(f"\nbest val accuracy {best:.4f} -> {a.out}")
    print("Next: python 03_calibrate.py   (fits per-type temperature on val)")
