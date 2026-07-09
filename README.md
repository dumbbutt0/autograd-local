# Autograd Foundations

Public learning artifacts for the frontier-eval/research-engineering track.

The goal is not to publish another polished tutorial. The goal is to show a real foundations trail:

1. implement scalar autograd from scratch,
2. verify gradients against PyTorch,
3. train a tiny MLP,
4. move into makemore and neural language-model foundations.

## Current artifacts

- `micrograd/clean_engine.py` - reusable scalar autograd engine and tiny MLP primitives.
- `micrograd/train_demo.py` - runnable toy MLP training demo.
- `micrograd/verify_against_torch.py` - checks a hand-built graph against PyTorch gradients.
- `makemore/neural_bigram.py` - neural bigram model: logits -> softmax -> NLL -> backward -> update.
- `public_posts/week1_micrograd.md` - short public post draft for GitHub/X.

## Run

```bash
python3 micrograd/verify_against_torch.py
python3 micrograd/train_demo.py
python3 makemore/neural_bigram.py
python3 -m unittest discover -s tests
```

## Claim ceiling

This repo currently proves only the first foundations step: I can build scalar backprop, train a tiny MLP, and verify a small graph against PyTorch. It does not claim transformer or RL competence yet.
