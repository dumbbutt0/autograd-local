#!/usr/bin/env python3
from __future__ import annotations

from clean_engine import MLP


xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0]


def main():
    model = MLP(3, [4, 4, 1], seed=42)
    lr = 0.05
    for step in range(80):
        ypred = [model(x) for x in xs]
        loss = sum((yout - ygt) ** 2 for ygt, yout in zip(ys, ypred))
        model.zero_grad()
        loss.backward()
        for p in model.parameters():
            p.data -= lr * p.grad
        if step % 10 == 0 or step == 79:
            print(f"step={step:02d} loss={loss.data:.6f}")
    print("predictions:", [round(model(x).data, 3) for x in xs])


if __name__ == "__main__":
    main()
