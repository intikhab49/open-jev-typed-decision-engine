"""Step 4 - fine-tune. Colab free T4 or HF Spaces; ~15 min for 10k examples.

  python 04_train.py --epochs 3 --max-len 1024
"""
import argparse, json, math
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import get_linear_schedule_with_warmup
from model import AgentDecisionEngine, get_tokenizer, DEFAULT_ENCODER
from data import load, split, targets


class States(Dataset):
    def __init__(self, rows, tok, max_len):
        self.rows, self.tok, self.max_len = rows, tok, max_len
        self.y = targets(rows)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        enc = self.tok(self.rows[i]["text"], truncation=True,
                       max_length=self.max_len, padding="max_length",
                       return_tensors="pt")
        return (enc["input_ids"][0], enc["attention_mask"][0],
                torch.tensor(self.y[0][i]), torch.tensor(self.y[1][i]),
                torch.tensor(self.y[2][i]))


def class_weights(y, n):
    c = np.bincount(y, minlength=n).astype("float32")
    w = c.sum() / (n * np.clip(c, 1, None))
    return torch.tensor(w)


def run_epoch(model, dl, dev, opt=None, sched=None, scaler=None, w_action=None):
    train = opt is not None
    model.train(train)
    tot, correct_a, n = 0.0, 0, 0
    for ids, mask, ye, ya, ys in dl:
        ids, mask = ids.to(dev), mask.to(dev)
        ye, ya, ys = ye.to(dev), ya.to(dev), ys.to(dev)
        with torch.autocast("cuda", dtype=torch.float16, enabled=(dev == "cuda")):
            le, la, ls = model(ids, mask)
            loss = (F.binary_cross_entropy_with_logits(le.float(), ye)
                    + F.cross_entropy(la.float(), ya, weight=w_action)
                    + 0.5 * F.cross_entropy(ls.float(), ys))
        if train:
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
        tot += loss.item() * len(ya)
        correct_a += (la.argmax(-1) == ya).sum().item()
        n += len(ya)
    return tot / n, correct_a / n


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--encoder", default=DEFAULT_ENCODER)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--bs", type=int, default=8)
    p.add_argument("--max-len", type=int, default=1024)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--head-lr", type=float, default=1e-3)
    p.add_argument("--out", default="engine.pt")
    a = p.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = get_tokenizer(a.encoder)
    s = split(load())
    dtr, dva = States(s["train"], tok, a.max_len), States(s["val"], tok, a.max_len)
    print(f"device {dev}  train {len(dtr)}  val {len(dva)}")

    model = AgentDecisionEngine(a.encoder).to(dev)
    w = class_weights(dtr.y[1], model.head_action.out_features).to(dev)

    head_params = [p_ for n_, p_ in model.named_parameters() if n_.startswith("head_")]
    enc_params = [p_ for n_, p_ in model.named_parameters() if not n_.startswith("head_")]
    opt = torch.optim.AdamW([{"params": enc_params, "lr": a.lr},
                             {"params": head_params, "lr": a.head_lr}], weight_decay=0.01)
    tl = DataLoader(dtr, batch_size=a.bs, shuffle=True, drop_last=True)
    vl = DataLoader(dva, batch_size=a.bs)
    steps = len(tl) * a.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=(dev == "cuda"))

    best = 0.0
    for e in range(a.epochs):
        trl, tra = run_epoch(model, tl, dev, opt, sched, scaler, w)
        with torch.no_grad():
            val, vaa = run_epoch(model, vl, dev, w_action=w)
        print(f"epoch {e+1}  train loss {trl:.3f} acc {tra:.3f} | "
              f"val loss {val:.3f} action acc {vaa:.3f}")
        if vaa > best:
            best = vaa
            torch.save({"state_dict": model.state_dict(), "encoder": a.encoder,
                        "max_len": a.max_len}, a.out)
            print(f"  saved -> {a.out}")
    print(f"best val action acc {best:.3f}")
