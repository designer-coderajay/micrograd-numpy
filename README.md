# MicroGrad-NumPy

**A scalar-value autograd engine built from scratch.**

Built without PyTorch, TensorFlow, or any autograd library. Just NumPy and math.

## What This Is

This is a fully functional automatic differentiation engine that implements backpropagation from first principles. No PyTorch, no TensorFlow, no autograd libraries. Just NumPy and math.

If you understand this code, you understand how deep learning frameworks work under the hood.

## Core Concepts

### The Value Class

Every computation in neural networks is built from simple operations. The `Value` class wraps a scalar and tracks:

1. **data** - The actual number
2. **grad** - The derivative of the final output with respect to this value
3. **_backward** - A function that propagates gradients to parent nodes
4. **_prev** - The values that produced this one (the computation graph)

```python
from micrograd_numpy import Value

a = Value(2.0, label='a')
b = Value(3.0, label='b')
c = a * b + a  # c = 2*3 + 2 = 8
c.backward()

print(f"dc/da = {a.grad}")  # dc/da = b + 1 = 4
print(f"dc/db = {b.grad}")  # dc/db = a = 2
```

### How Backpropagation Works

1. **Forward pass**: Build the computation graph by executing operations
2. **Backward pass**: Walk the graph in reverse topological order, applying chain rule

The chain rule says: if `z = f(y)` and `y = g(x)`, then `dz/dx = dz/dy * dy/dx`

Each operation knows its local derivative. Backprop chains them together.

## Supported Operations

| Operation | Forward | Backward (local derivative) |
|-----------|---------|----------------------------|
| `a + b` | `a.data + b.data` | `∂/∂a = 1, ∂/∂b = 1` |
| `a * b` | `a.data * b.data` | `∂/∂a = b, ∂/∂b = a` |
| `a ** n` | `a.data ** n` | `∂/∂a = n * a^(n-1)` |
| `a.relu()` | `max(0, a)` | `∂/∂a = 1 if a > 0 else 0` |
| `a.tanh()` | `tanh(a)` | `∂/∂a = 1 - tanh(a)²` |
| `a.sigmoid()` | `1/(1+e^-a)` | `∂/∂a = σ(a) * (1 - σ(a))` |
| `a.exp()` | `e^a` | `∂/∂a = e^a` |
| `a.log()` | `ln(a)` | `∂/∂a = 1/a` |

## Neural Network API

The API mirrors PyTorch:

```python
from micrograd_numpy import MLP, SGD, mse_loss, Value

# Create model: 2 inputs → 4 hidden → 4 hidden → 1 output
model = MLP(2, [4, 4, 1])

# Training loop
optimizer = SGD(model.parameters(), lr=0.1)

for epoch in range(100):
    # Forward pass
    predictions = [model([Value(x[0]), Value(x[1])]) for x in X]
    loss = mse_loss(predictions, y)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

## Project Structure

```
micrograd_numpy/
├── __init__.py      # Public API exports
├── engine.py        # Value class, backprop, computation graph
└── nn.py            # Neuron, Layer, MLP, loss functions, optimizers

tests/
└── test_engine.py   # Unit tests comparing against PyTorch

examples/
└── demo.py          # Training demo on moons dataset
```

## Running Tests

```bash
# Run all tests
pytest tests/test_engine.py -v

# Run only PyTorch comparison tests
pytest tests/test_engine.py -v -k "PyTorch"

# Run with coverage
pytest tests/test_engine.py --cov=micrograd_numpy
```

## Running the Demo

```bash
python examples/demo.py
```

This will:
1. Demonstrate gradient computation
2. Show the computation graph
3. Train a neural network on the moons dataset
4. Generate decision boundary and loss curve plots

## Key Files Explained

### `engine.py`

The heart of the system. Contains:

- **Value**: The autograd-enabled scalar
- **Arithmetic operators**: `__add__`, `__mul__`, `__pow__`, etc.
- **Activations**: `relu()`, `tanh()`, `sigmoid()`, `exp()`, `log()`
- **backward()**: Topological sort + chain rule application
- **topological_sort()**: DFS-based graph ordering
- **draw_graph()**: Visualization helper

### `nn.py`

Neural network building blocks:

- **Module**: Base class with `parameters()` and `zero_grad()`
- **Neuron**: Single neuron with weights, bias, activation
- **Layer**: Collection of neurons (fully connected)
- **MLP**: Multi-layer perceptron
- **Loss functions**: `mse_loss`, `binary_cross_entropy`, `hinge_loss`
- **Optimizers**: `SGD`, `Adam`

## Why This Matters for Senior Engineers

1. **Framework Independence**: You can debug PyTorch/TensorFlow because you know what they're doing
2. **Custom Operations**: Need a weird activation? You know how to make it differentiable
3. **Performance Understanding**: You know why certain operations are slow
4. **Research**: Implementing papers requires understanding gradients

## The Math

The entire engine is built on one equation - the chain rule:

```
∂L/∂x = ∂L/∂y * ∂y/∂x
```

For any output `L` and intermediate value `y = f(x)`:
- `∂L/∂y` comes from later in the graph (it's `y.grad`)
- `∂y/∂x` is the local derivative (we compute it based on the operation)
- `∂L/∂x` accumulates into `x.grad`

That's it. Everything else is bookkeeping.

## License

MIT. Build something cool.
