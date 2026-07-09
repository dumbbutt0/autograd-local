#!/usr/bin/env python3
from __future__ import annotations

from clean_engine import Value


def micrograd_result():
    x1 = Value(2.0, label="x1")
    x2 = Value(0.0, label="x2")
    w1 = Value(-3.0, label="w1")
    w2 = Value(1.0, label="w2")
    b = Value(6.881373587870195, label="b")
    n = x1 * w1 + x2 * w2 + b
    o = n.tanh()
    o.backward()
    return o.data, {"x1": x1.grad, "x2": x2.grad, "w1": w1.grad, "w2": w2.grad, "b": b.grad}


def torch_result():
    import torch

    x1 = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)
    x2 = torch.tensor([0.0], dtype=torch.float64, requires_grad=True)
    w1 = torch.tensor([-3.0], dtype=torch.float64, requires_grad=True)
    w2 = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)
    b = torch.tensor([6.881373587870195], dtype=torch.float64, requires_grad=True)
    o = torch.tanh(x1 * w1 + x2 * w2 + b)
    o.backward()
    return o.item(), {"x1": x1.grad.item(), "x2": x2.grad.item(), "w1": w1.grad.item(), "w2": w2.grad.item(), "b": b.grad.item()}


def main():
    m_val, m_grads = micrograd_result()
    t_val, t_grads = torch_result()
    assert abs(m_val - t_val) < 1e-12
    for key in m_grads:
        assert abs(m_grads[key] - t_grads[key]) < 1e-12, (key, m_grads[key], t_grads[key])
    print("micrograd matches PyTorch")
    print(f"output={m_val:.12f}")
    for key, value in m_grads.items():
        print(f"{key}.grad={value:.12f}")


if __name__ == "__main__":
    main()
