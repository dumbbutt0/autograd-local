# Week 1: Scalar Autograd From Scratch

Claim ceiling: I am not claiming broad ML competence yet. This artifact shows that I can build the smallest real version of backprop and verify it against PyTorch.

What I built:
- a scalar `Value` object that records a computation graph,
- local backward rules for `+`, `-`, `*`, `/`, powers, `exp`, and `tanh`,
- reverse topological backprop,
- a tiny `Neuron -> Layer -> MLP` stack,
- a toy training loop that drives loss down.

What I verified:
- the classic `tanh(w*x+b)` graph matches PyTorch gradients to numerical precision,
- the `x + x` gradient accumulation bug is covered by a test,
- the tiny MLP loss decreases on a 4-example toy dataset.

What surprised me:
- backprop is not magic; it is local derivatives plus careful graph traversal,
- forgetting to zero gradients is a real failure mode immediately,
- the boring `+=` vs `=` detail matters because graph nodes can be reused.

Next:
- finish makemore's neural bigram model,
- move from scalar autograd to PyTorch tensors, softmax, NLL, and embeddings.
