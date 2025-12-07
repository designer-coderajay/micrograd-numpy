"""
Unit Tests: MicroGrad vs PyTorch
================================

The ultimate validation: our gradients must match PyTorch's gradients.

If these tests pass, we've built a correct autograd engine. PyTorch is the
ground truth, and we verify that our implementation produces identical
results for every operation.

Test coverage:
- Basic arithmetic (add, mul, sub, div, pow)
- Activation functions (relu, tanh, sigmoid)
- Composite expressions
- Neural network forward/backward
- Edge cases and numerical stability

Run with: pytest tests/test_engine.py -v
"""

import sys
import math
import pytest
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, '/home/claude')

from micrograd_numpy import Value, MLP, Neuron, Layer, mse_loss, SGD


# Try to import PyTorch for comparison tests
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Skipping comparison tests.")


# =============================================================================
# Test Configuration
# =============================================================================

TOLERANCE = 1e-6  # Acceptable difference between our grads and PyTorch's


def assert_close(actual: float, expected: float, tol: float = TOLERANCE) -> None:
    """Assert two values are approximately equal."""
    diff = abs(actual - expected)
    assert diff < tol, f"Values differ: {actual} vs {expected} (diff={diff})"


# =============================================================================
# Basic Value Tests (No PyTorch Required)
# =============================================================================

class TestValueBasics:
    """Test basic Value functionality without PyTorch comparison."""

    def test_value_creation(self) -> None:
        """Test that Values store data correctly."""
        v = Value(3.14)
        assert v.data == 3.14
        assert v.grad == 0.0

    def test_value_with_label(self) -> None:
        """Test labeled Values."""
        v = Value(2.0, label='x')
        assert v.label == 'x'
        assert 'x' in repr(v)

    def test_addition(self) -> None:
        """Test addition produces correct forward value."""
        a = Value(2.0)
        b = Value(3.0)
        c = a + b
        assert c.data == 5.0

    def test_multiplication(self) -> None:
        """Test multiplication produces correct forward value."""
        a = Value(2.0)
        b = Value(3.0)
        c = a * b
        assert c.data == 6.0

    def test_subtraction(self) -> None:
        """Test subtraction produces correct forward value."""
        a = Value(5.0)
        b = Value(3.0)
        c = a - b
        assert c.data == 2.0

    def test_division(self) -> None:
        """Test division produces correct forward value."""
        a = Value(6.0)
        b = Value(2.0)
        c = a / b
        assert c.data == 3.0

    def test_power(self) -> None:
        """Test power produces correct forward value."""
        a = Value(2.0)
        c = a ** 3
        assert c.data == 8.0

    def test_negation(self) -> None:
        """Test negation."""
        a = Value(5.0)
        b = -a
        assert b.data == -5.0

    def test_relu_positive(self) -> None:
        """Test ReLU on positive input."""
        a = Value(3.0)
        b = a.relu()
        assert b.data == 3.0

    def test_relu_negative(self) -> None:
        """Test ReLU on negative input."""
        a = Value(-3.0)
        b = a.relu()
        assert b.data == 0.0

    def test_tanh(self) -> None:
        """Test tanh activation."""
        a = Value(0.5)
        b = a.tanh()
        assert_close(b.data, math.tanh(0.5))

    def test_sigmoid(self) -> None:
        """Test sigmoid activation."""
        a = Value(0.5)
        b = a.sigmoid()
        expected = 1 / (1 + math.exp(-0.5))
        assert_close(b.data, expected)

    def test_exp(self) -> None:
        """Test exponential."""
        a = Value(2.0)
        b = a.exp()
        assert_close(b.data, math.exp(2.0))

    def test_log(self) -> None:
        """Test natural log."""
        a = Value(2.0)
        b = a.log()
        assert_close(b.data, math.log(2.0))

    def test_radd(self) -> None:
        """Test reverse addition (number + Value)."""
        a = Value(2.0)
        b = 3 + a
        assert b.data == 5.0

    def test_rmul(self) -> None:
        """Test reverse multiplication (number * Value)."""
        a = Value(2.0)
        b = 3 * a
        assert b.data == 6.0


class TestBackwardBasics:
    """Test backward pass without PyTorch comparison."""

    def test_simple_add_backward(self) -> None:
        """Test gradient flow through addition."""
        a = Value(2.0)
        b = Value(3.0)
        c = a + b
        c.backward()

        # dc/da = 1, dc/db = 1
        assert a.grad == 1.0
        assert b.grad == 1.0

    def test_simple_mul_backward(self) -> None:
        """Test gradient flow through multiplication."""
        a = Value(2.0)
        b = Value(3.0)
        c = a * b
        c.backward()

        # dc/da = b = 3, dc/db = a = 2
        assert a.grad == 3.0
        assert b.grad == 2.0

    def test_chain_rule(self) -> None:
        """Test chain rule: d(f(g(x)))/dx = f'(g(x)) * g'(x)."""
        x = Value(2.0)
        y = x * x  # y = x^2
        z = y * y  # z = y^2 = x^4
        z.backward()

        # dz/dx = 4x^3 = 4 * 8 = 32
        assert x.grad == 32.0

    def test_diamond_graph(self) -> None:
        """Test gradient accumulation in diamond-shaped graph."""
        a = Value(2.0)
        b = a + a  # Uses 'a' twice
        b.backward()

        # db/da = 2 (gradient accumulates from both paths)
        assert a.grad == 2.0

    def test_power_backward(self) -> None:
        """Test gradient of power function."""
        x = Value(3.0)
        y = x ** 2
        y.backward()

        # dy/dx = 2x = 6
        assert x.grad == 6.0

    def test_relu_backward_positive(self) -> None:
        """Test ReLU gradient for positive input."""
        x = Value(2.0)
        y = x.relu()
        y.backward()

        # ReLU passes gradient through for positive inputs
        assert x.grad == 1.0

    def test_relu_backward_negative(self) -> None:
        """Test ReLU gradient for negative input."""
        x = Value(-2.0)
        y = x.relu()
        y.backward()

        # ReLU blocks gradient for negative inputs
        assert x.grad == 0.0


# =============================================================================
# PyTorch Comparison Tests
# =============================================================================

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
class TestPyTorchComparison:
    """Compare our gradients against PyTorch's gradients."""

    def test_add_grad(self) -> None:
        """Compare addition gradients."""
        # MicroGrad
        a = Value(2.0)
        b = Value(3.0)
        c = a + b
        c.backward()

        # PyTorch
        a_t = torch.tensor(2.0, requires_grad=True)
        b_t = torch.tensor(3.0, requires_grad=True)
        c_t = a_t + b_t
        c_t.backward()

        assert_close(a.grad, a_t.grad.item())
        assert_close(b.grad, b_t.grad.item())

    def test_mul_grad(self) -> None:
        """Compare multiplication gradients."""
        a = Value(2.0)
        b = Value(3.0)
        c = a * b
        c.backward()

        a_t = torch.tensor(2.0, requires_grad=True)
        b_t = torch.tensor(3.0, requires_grad=True)
        c_t = a_t * b_t
        c_t.backward()

        assert_close(a.grad, a_t.grad.item())
        assert_close(b.grad, b_t.grad.item())

    def test_sub_grad(self) -> None:
        """Compare subtraction gradients."""
        a = Value(5.0)
        b = Value(3.0)
        c = a - b
        c.backward()

        a_t = torch.tensor(5.0, requires_grad=True)
        b_t = torch.tensor(3.0, requires_grad=True)
        c_t = a_t - b_t
        c_t.backward()

        assert_close(a.grad, a_t.grad.item())
        assert_close(b.grad, b_t.grad.item())

    def test_div_grad(self) -> None:
        """Compare division gradients."""
        a = Value(6.0)
        b = Value(2.0)
        c = a / b
        c.backward()

        a_t = torch.tensor(6.0, requires_grad=True)
        b_t = torch.tensor(2.0, requires_grad=True)
        c_t = a_t / b_t
        c_t.backward()

        assert_close(a.grad, a_t.grad.item())
        assert_close(b.grad, b_t.grad.item())

    def test_pow_grad(self) -> None:
        """Compare power gradients."""
        a = Value(2.0)
        c = a ** 3
        c.backward()

        a_t = torch.tensor(2.0, requires_grad=True)
        c_t = a_t ** 3
        c_t.backward()

        assert_close(a.grad, a_t.grad.item())

    def test_neg_grad(self) -> None:
        """Compare negation gradients."""
        a = Value(3.0)
        c = -a
        c.backward()

        a_t = torch.tensor(3.0, requires_grad=True)
        c_t = -a_t
        c_t.backward()

        assert_close(a.grad, a_t.grad.item())

    def test_tanh_grad(self) -> None:
        """Compare tanh gradients."""
        a = Value(0.5)
        c = a.tanh()
        c.backward()

        a_t = torch.tensor(0.5, requires_grad=True)
        c_t = torch.tanh(a_t)
        c_t.backward()

        assert_close(c.data, c_t.item())
        assert_close(a.grad, a_t.grad.item())

    def test_relu_grad_positive(self) -> None:
        """Compare ReLU gradients for positive input."""
        a = Value(2.0)
        c = a.relu()
        c.backward()

        a_t = torch.tensor(2.0, requires_grad=True)
        c_t = torch.relu(a_t)
        c_t.backward()

        assert_close(c.data, c_t.item())
        assert_close(a.grad, a_t.grad.item())

    def test_relu_grad_negative(self) -> None:
        """Compare ReLU gradients for negative input."""
        a = Value(-2.0)
        c = a.relu()
        c.backward()

        a_t = torch.tensor(-2.0, requires_grad=True)
        c_t = torch.relu(a_t)
        c_t.backward()

        assert_close(c.data, c_t.item())
        assert_close(a.grad, a_t.grad.item())

    def test_sigmoid_grad(self) -> None:
        """Compare sigmoid gradients."""
        a = Value(0.5)
        c = a.sigmoid()
        c.backward()

        a_t = torch.tensor(0.5, requires_grad=True)
        c_t = torch.sigmoid(a_t)
        c_t.backward()

        assert_close(c.data, c_t.item())
        assert_close(a.grad, a_t.grad.item())

    def test_exp_grad(self) -> None:
        """Compare exp gradients."""
        a = Value(1.5)
        c = a.exp()
        c.backward()

        a_t = torch.tensor(1.5, requires_grad=True)
        c_t = torch.exp(a_t)
        c_t.backward()

        assert_close(c.data, c_t.item())
        assert_close(a.grad, a_t.grad.item())

    def test_log_grad(self) -> None:
        """Compare log gradients."""
        a = Value(2.0)
        c = a.log()
        c.backward()

        a_t = torch.tensor(2.0, requires_grad=True)
        c_t = torch.log(a_t)
        c_t.backward()

        assert_close(c.data, c_t.item())
        assert_close(a.grad, a_t.grad.item())

    def test_complex_expression_1(self) -> None:
        """Compare gradients for: (a + b) * (b + 1)."""
        a = Value(2.0)
        b = Value(3.0)
        c = (a + b) * (b + 1)
        c.backward()

        a_t = torch.tensor(2.0, requires_grad=True)
        b_t = torch.tensor(3.0, requires_grad=True)
        c_t = (a_t + b_t) * (b_t + 1)
        c_t.backward()

        assert_close(c.data, c_t.item())
        assert_close(a.grad, a_t.grad.item())
        assert_close(b.grad, b_t.grad.item())

    def test_complex_expression_2(self) -> None:
        """Compare gradients for: tanh(a * b + a^2)."""
        a = Value(1.0)
        b = Value(2.0)
        c = (a * b + a ** 2).tanh()
        c.backward()

        a_t = torch.tensor(1.0, requires_grad=True)
        b_t = torch.tensor(2.0, requires_grad=True)
        c_t = torch.tanh(a_t * b_t + a_t ** 2)
        c_t.backward()

        assert_close(c.data, c_t.item())
        assert_close(a.grad, a_t.grad.item())
        assert_close(b.grad, b_t.grad.item())

    def test_complex_expression_3(self) -> None:
        """Compare gradients for neuron-like computation."""
        # Simulate: relu(w1*x1 + w2*x2 + b)
        w1 = Value(0.5)
        w2 = Value(-0.3)
        x1 = Value(2.0)
        x2 = Value(3.0)
        b = Value(0.1)

        out = (w1 * x1 + w2 * x2 + b).relu()
        out.backward()

        w1_t = torch.tensor(0.5, requires_grad=True)
        w2_t = torch.tensor(-0.3, requires_grad=True)
        x1_t = torch.tensor(2.0, requires_grad=True)
        x2_t = torch.tensor(3.0, requires_grad=True)
        b_t = torch.tensor(0.1, requires_grad=True)

        out_t = torch.relu(w1_t * x1_t + w2_t * x2_t + b_t)
        out_t.backward()

        assert_close(out.data, out_t.item())
        assert_close(w1.grad, w1_t.grad.item())
        assert_close(w2.grad, w2_t.grad.item())
        assert_close(b.grad, b_t.grad.item())

    def test_repeated_variable(self) -> None:
        """Compare gradients when variable is used multiple times."""
        x = Value(3.0)
        y = x * x * x  # x^3
        y.backward()

        x_t = torch.tensor(3.0, requires_grad=True)
        y_t = x_t * x_t * x_t
        y_t.backward()

        assert_close(y.data, y_t.item())
        assert_close(x.grad, x_t.grad.item())

    def test_long_chain(self) -> None:
        """Compare gradients for long computation chain."""
        a = Value(0.5)
        b = a
        for _ in range(10):
            b = b * a + a
        b.backward()

        a_t = torch.tensor(0.5, requires_grad=True)
        b_t = a_t
        for _ in range(10):
            b_t = b_t * a_t + a_t
        b_t.backward()

        assert_close(b.data, b_t.item())
        assert_close(a.grad, a_t.grad.item(), tol=1e-4)


# =============================================================================
# Neural Network Tests
# =============================================================================

class TestNeuralNetworks:
    """Test neural network components."""

    def test_neuron_creation(self) -> None:
        """Test neuron initialization."""
        n = Neuron(3)
        assert len(n.w) == 3
        assert n.b is not None

    def test_neuron_forward(self) -> None:
        """Test neuron forward pass."""
        n = Neuron(2, nonlin=False)  # Linear for predictable output
        n.w[0].data = 1.0
        n.w[1].data = 2.0
        n.b.data = 0.5

        out = n([Value(1.0), Value(1.0)])
        # 1*1 + 2*1 + 0.5 = 3.5
        assert out.data == 3.5

    def test_neuron_parameters(self) -> None:
        """Test neuron parameter collection."""
        n = Neuron(3)
        params = n.parameters()
        assert len(params) == 4  # 3 weights + 1 bias

    def test_layer_creation(self) -> None:
        """Test layer initialization."""
        layer = Layer(3, 4)
        assert len(layer.neurons) == 4

    def test_layer_forward(self) -> None:
        """Test layer forward pass."""
        layer = Layer(2, 3)
        x = [Value(1.0), Value(1.0)]
        out = layer(x)
        assert len(out) == 3

    def test_layer_parameters(self) -> None:
        """Test layer parameter collection."""
        layer = Layer(2, 3)
        params = layer.parameters()
        # 3 neurons * (2 weights + 1 bias) = 9 parameters
        assert len(params) == 9

    def test_mlp_creation(self) -> None:
        """Test MLP initialization."""
        mlp = MLP(3, [4, 4, 1])
        assert len(mlp.layers) == 3

    def test_mlp_forward(self) -> None:
        """Test MLP forward pass."""
        mlp = MLP(2, [4, 1])
        x = [Value(1.0), Value(2.0)]
        out = mlp(x)
        assert isinstance(out, Value)

    def test_mlp_parameters(self) -> None:
        """Test MLP parameter collection."""
        mlp = MLP(2, [3, 1])
        params = mlp.parameters()
        # Layer 1: 3 * (2 + 1) = 9
        # Layer 2: 1 * (3 + 1) = 4
        # Total = 13
        assert len(params) == 13

    def test_mlp_backward(self) -> None:
        """Test MLP backward pass computes gradients."""
        mlp = MLP(2, [3, 1])
        x = [Value(1.0), Value(2.0)]
        out = mlp(x)
        out.backward()

        # All parameters should have non-zero gradients (probably)
        params = mlp.parameters()
        has_nonzero = any(p.grad != 0.0 for p in params)
        assert has_nonzero

    def test_zero_grad(self) -> None:
        """Test gradient zeroing."""
        mlp = MLP(2, [3, 1])
        x = [Value(1.0), Value(2.0)]

        # First backward
        out = mlp(x)
        out.backward()

        # Zero gradients
        mlp.zero_grad()

        # All gradients should be zero
        for p in mlp.parameters():
            assert p.grad == 0.0


class TestTraining:
    """Test training loop components."""

    def test_mse_loss(self) -> None:
        """Test MSE loss computation."""
        preds = [Value(1.0), Value(2.0), Value(3.0)]
        targets = [1.0, 2.0, 3.0]
        loss = mse_loss(preds, targets)
        assert loss.data == 0.0  # Perfect predictions

        preds = [Value(0.0), Value(0.0), Value(0.0)]
        loss = mse_loss(preds, targets)
        # MSE = (1 + 4 + 9) / 3 = 14/3
        assert_close(loss.data, 14 / 3)

    def test_sgd_step(self) -> None:
        """Test SGD optimizer step."""
        w = Value(1.0)
        w.grad = 0.1

        optimizer = SGD([w], lr=0.1)
        optimizer.step()

        # w = w - lr * grad = 1.0 - 0.1 * 0.1 = 0.99
        assert_close(w.data, 0.99)

    def test_training_reduces_loss(self) -> None:
        """Test that training reduces loss over time."""
        np.random.seed(42)

        # Simple XOR-like problem
        X = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
        y = [0.0, 1.0, 1.0, 0.0]

        mlp = MLP(2, [4, 1])
        optimizer = SGD(mlp.parameters(), lr=0.1)

        initial_loss = None
        final_loss = None

        for epoch in range(100):
            # Forward pass
            preds = [mlp([Value(x[0]), Value(x[1])]) for x in X]
            loss = mse_loss(preds, y)

            if epoch == 0:
                initial_loss = loss.data
            if epoch == 99:
                final_loss = loss.data

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Loss should decrease
        assert final_loss < initial_loss


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_zero_division_handling(self) -> None:
        """Test division by very small numbers."""
        a = Value(1.0)
        b = Value(1e-10)
        c = a / b
        assert c.data == 1e10

    def test_large_values(self) -> None:
        """Test with large values."""
        a = Value(1e10)
        b = Value(1e10)
        c = a + b
        assert c.data == 2e10

    def test_small_values(self) -> None:
        """Test with small values."""
        a = Value(1e-10)
        b = Value(1e-10)
        c = a * b
        assert_close(c.data, 1e-20)

    def test_negative_power(self) -> None:
        """Test negative power."""
        a = Value(2.0)
        c = a ** -1
        assert c.data == 0.5

    def test_fractional_power(self) -> None:
        """Test fractional power (square root)."""
        a = Value(4.0)
        c = a ** 0.5
        assert c.data == 2.0

    def test_type_error_on_invalid_data(self) -> None:
        """Test that invalid data types raise TypeError."""
        with pytest.raises(TypeError):
            Value("not a number")

    def test_log_negative_raises(self) -> None:
        """Test that log of negative raises ValueError."""
        a = Value(-1.0)
        with pytest.raises(ValueError):
            a.log()

    def test_neuron_input_mismatch(self) -> None:
        """Test that wrong input size raises ValueError."""
        n = Neuron(3)
        with pytest.raises(ValueError):
            n([Value(1.0), Value(2.0)])  # Only 2 inputs, expected 3


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
