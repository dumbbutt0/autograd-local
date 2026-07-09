#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
WORDS = (ROOT / "names.txt").read_text(encoding="utf-8").splitlines()
CHARS = sorted(set("".join(WORDS)))
STOI = {s: i + 1 for i, s in enumerate(CHARS)}
STOI["."] = 0
ITOS = {i: s for s, i in STOI.items()}
VOCAB_SIZE = len(STOI)


def build_dataset(words):
    xs, ys = [], []
    for word in words:
        chars = ["."] + list(word) + ["."]
        for ch1, ch2 in zip(chars, chars[1:]):
            xs.append(STOI[ch1])
            ys.append(STOI[ch2])
    return torch.tensor(xs), torch.tensor(ys)


def sample(W, generator, n=10):
    out = []
    for _ in range(n):
        ix = 0
        chars = []
        while True:
            xenc = F.one_hot(torch.tensor([ix]), num_classes=VOCAB_SIZE).float()
            logits = xenc @ W
            probs = logits.exp()
            probs = probs / probs.sum(1, keepdim=True)
            ix = torch.multinomial(probs[0], num_samples=1, replacement=True, generator=generator).item()
            if ix == 0:
                break
            chars.append(ITOS[ix])
        out.append("".join(chars))
    return out


def main():
    xs, ys = build_dataset(WORDS)
    xenc = F.one_hot(xs, num_classes=VOCAB_SIZE).float()
    g = torch.Generator().manual_seed(2147483647)
    W = torch.randn((VOCAB_SIZE, VOCAB_SIZE), generator=g, requires_grad=True)

    for step in range(120):
        logits = xenc @ W
        counts = logits.exp()
        probs = counts / counts.sum(1, keepdim=True)
        loss = -probs[torch.arange(xs.nelement()), ys].log().mean() + 0.01 * (W**2).mean()
        W.grad = None
        loss.backward()
        W.data += -50.0 * W.grad
        if step % 20 == 0 or step == 119:
            print(f"step={step:03d} loss={loss.item():.4f}")

    print("samples:")
    for name in sample(W, torch.Generator().manual_seed(42), n=20):
        print(name)


if __name__ == "__main__":
    main()
