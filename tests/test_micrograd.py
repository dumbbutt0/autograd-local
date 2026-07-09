import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "micrograd"))

from clean_engine import Value, MLP  # noqa: E402


class MicrogradTests(unittest.TestCase):
    def test_gradient_accumulates_for_reused_node(self):
        a = Value(3.0)
        b = a + a
        b.backward()
        self.assertEqual(a.grad, 2.0)

    def test_toy_mlp_loss_decreases(self):
        xs = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]]
        ys = [1.0, -1.0, -1.0, 1.0]
        model = MLP(3, [4, 4, 1], seed=123)

        def loss_value():
            ypred = [model(x) for x in xs]
            return sum((yout - ygt) ** 2 for ygt, yout in zip(ys, ypred))

        initial = loss_value().data
        for _ in range(40):
            loss = loss_value()
            model.zero_grad()
            loss.backward()
            for p in model.parameters():
                p.data -= 0.05 * p.grad
        final = loss_value().data
        self.assertLess(final, initial)


if __name__ == "__main__":
    unittest.main()
