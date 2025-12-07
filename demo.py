#!/usr/bin/env python3
"""
MicroGrad Demo: Training a Neural Network from Scratch
=======================================================

This demo shows the complete workflow:
1. Create a dataset (moons classification problem)
2. Build a neural network using our autograd engine
3. Train it with gradient descent
4. Visualize the decision boundary

No PyTorch. No TensorFlow. Just our autograd engine and NumPy.

Run: python examples/demo.py
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

# Add parent directory to path
sys.path.insert(0, '.')

from micrograd_numpy import Value, MLP, SGD, Adam, hinge_loss


def make_moons(
    n_samples: int = 100,
    noise: float = 0.1,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate the classic 'moons' dataset for binary classification.

    Two interleaved half-circles that are not linearly separable.
    Perfect for demonstrating neural network capabilities.

    Args:
        n_samples: Total number of samples.
        noise: Standard deviation of Gaussian noise.
        seed: Random seed for reproducibility.

    Returns:
        X: Features array of shape (n_samples, 2)
        y: Labels array of shape (n_samples,) with values -1 or 1
    """
    np.random.seed(seed)

    n_each = n_samples // 2

    # First moon (top)
    theta1 = np.linspace(0, np.pi, n_each)
    x1 = np.cos(theta1)
    y1 = np.sin(theta1)

    # Second moon (bottom, shifted)
    theta2 = np.linspace(0, np.pi, n_each)
    x2 = 1 - np.cos(theta2)
    y2 = 0.5 - np.sin(theta2)

    # Combine
    X = np.vstack([
        np.column_stack([x1, y1]),
        np.column_stack([x2, y2])
    ])

    # Add noise
    X += np.random.randn(*X.shape) * noise

    # Labels: 1 for first moon, -1 for second
    y = np.array([1] * n_each + [-1] * n_each)

    return X, y


def accuracy(model: MLP, X: np.ndarray, y: np.ndarray) -> float:
    """
    Compute classification accuracy.

    Args:
        model: Trained MLP model.
        X: Feature matrix.
        y: True labels (-1 or 1).

    Returns:
        Accuracy as a float between 0 and 1.
    """
    correct = 0
    for xi, yi in zip(X, y):
        pred = model([Value(xi[0]), Value(xi[1])])
        pred_class = 1 if pred.data > 0 else -1
        if pred_class == yi:
            correct += 1
    return correct / len(y)


def train(
    model: MLP,
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = 100,
    lr: float = 0.1,
    verbose: bool = True
) -> List[float]:
    """
    Train the model using gradient descent.

    Args:
        model: MLP to train.
        X: Training features.
        y: Training labels.
        epochs: Number of training iterations.
        lr: Learning rate.
        verbose: Whether to print progress.

    Returns:
        List of loss values per epoch.
    """
    optimizer = SGD(model.parameters(), lr=lr)
    losses = []

    for epoch in range(epochs):
        # Forward pass: compute predictions
        predictions = []
        for xi in X:
            pred = model([Value(xi[0]), Value(xi[1])])
            predictions.append(pred)

        # Compute loss (SVM-style hinge loss)
        # Add L2 regularization for better generalization
        data_loss = hinge_loss(predictions, y.tolist())
        reg_loss = sum(p.data ** 2 for p in model.parameters()) * 0.0001
        total_loss = data_loss + reg_loss

        losses.append(total_loss.data)

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()

        # Update parameters
        optimizer.step()

        if verbose and (epoch + 1) % 10 == 0:
            acc = accuracy(model, X, y)
            print(f"Epoch {epoch + 1:3d} | Loss: {total_loss.data:.4f} | Accuracy: {acc:.2%}")

    return losses


def plot_decision_boundary(
    model: MLP,
    X: np.ndarray,
    y: np.ndarray,
    title: str = "Decision Boundary"
) -> None:
    """
    Visualize the decision boundary learned by the model.

    Args:
        model: Trained MLP.
        X: Feature matrix.
        y: Labels.
        title: Plot title.
    """
    # Create mesh grid
    h = 0.02  # Step size
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h),
        np.arange(y_min, y_max, h)
    )

    # Predict on mesh
    Z = []
    for x1, x2 in zip(xx.ravel(), yy.ravel()):
        pred = model([Value(x1), Value(x2)])
        Z.append(pred.data)
    Z = np.array(Z).reshape(xx.shape)

    # Plot
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, levels=50, cmap='RdBu', alpha=0.8)
    plt.colorbar(label='Model output')
    plt.contour(xx, yy, Z, levels=[0], colors='black', linewidths=2)

    # Plot data points
    scatter = plt.scatter(
        X[:, 0], X[:, 1],
        c=y,
        cmap='RdBu',
        edgecolors='black',
        s=50
    )

    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title(title)
    plt.tight_layout()
    plt.savefig('./decision_boundary.png', dpi=150)
    plt.close()
    print("Saved decision boundary plot to: decision_boundary.png")


def plot_loss_curve(losses: List[float]) -> None:
    """
    Plot the training loss over epochs.

    Args:
        losses: List of loss values.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(losses, 'b-', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('./loss_curve.png', dpi=150)
    plt.close()
    print("Saved loss curve to: loss_curve.png")


def demo_gradient_computation():
    """
    Demonstrate basic gradient computation.

    Shows how our engine computes derivatives automatically.
    """
    print("=" * 60)
    print("DEMO 1: Automatic Gradient Computation")
    print("=" * 60)
    print()

    # Simple example: f(x) = x^2 + 2x + 1
    print("Computing gradients for f(x) = x² + 2x + 1 at x = 3")
    print()

    x = Value(3.0, label='x')
    f = x ** 2 + 2 * x + 1
    f.backward()

    print(f"f(3) = {f.data}")
    print(f"df/dx at x=3 = {x.grad}")
    print(f"(Analytical: df/dx = 2x + 2 = 2(3) + 2 = 8)")
    print()

    # More complex example
    print("Computing gradients for g(a,b) = tanh(a*b + a²)")
    print()

    a = Value(2.0, label='a')
    b = Value(3.0, label='b')
    g = (a * b + a ** 2).tanh()
    g.backward()

    print(f"g(2, 3) = tanh(2*3 + 4) = tanh(10) = {g.data:.6f}")
    print(f"dg/da = {a.grad:.6f}")
    print(f"dg/db = {b.grad:.6f}")
    print()


def demo_neural_network():
    """
    Train a neural network on the moons dataset.
    """
    print("=" * 60)
    print("DEMO 2: Training a Neural Network")
    print("=" * 60)
    print()

    # Generate data
    print("Generating moons dataset (100 samples)...")
    X, y = make_moons(n_samples=100, noise=0.15)
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    print()

    # Create model
    print("Creating MLP: 2 inputs → 16 hidden → 16 hidden → 1 output")
    model = MLP(2, [16, 16, 1])
    n_params = len(model.parameters())
    print(f"Total parameters: {n_params}")
    print()

    # Train
    print("Training for 100 epochs...")
    print("-" * 40)
    losses = train(model, X, y, epochs=100, lr=0.5, verbose=True)
    print("-" * 40)
    print()

    # Final accuracy
    final_acc = accuracy(model, X, y)
    print(f"Final training accuracy: {final_acc:.2%}")
    print()

    # Plot results
    print("Generating visualizations...")
    plot_decision_boundary(model, X, y, f"Decision Boundary (Accuracy: {final_acc:.1%})")
    plot_loss_curve(losses)
    print()


def demo_backprop_visualization():
    """
    Show the computation graph and gradient flow.
    """
    print("=" * 60)
    print("DEMO 3: Computation Graph Visualization")
    print("=" * 60)
    print()

    from micrograd_numpy import draw_graph

    # Build a small computation
    x = Value(2.0, label='x')
    y = Value(3.0, label='y')
    z = x * y
    z.label = 'z=x*y'
    w = z + x
    w.label = 'w=z+x'
    out = w.tanh()
    out.label = 'out=tanh(w)'

    out.backward()

    print("Expression: out = tanh(x*y + x)")
    print(f"At x=2, y=3:")
    print(f"  z = x*y = 6")
    print(f"  w = z+x = 8")
    print(f"  out = tanh(8) = {out.data:.6f}")
    print()
    print("Gradients (via backpropagation):")
    print(f"  d(out)/dx = {x.grad:.6f}")
    print(f"  d(out)/dy = {y.grad:.6f}")
    print()
    print("Computation Graph (text format):")
    print(draw_graph(out, format='text'))
    print()


def main():
    """Run all demos."""
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║           MicroGrad-NumPy: Autograd Engine Demo          ║")
    print("║                                                          ║")
    print("║   Built from scratch. No PyTorch. No TensorFlow.         ║")
    print("║   Just pure math and the chain rule.                     ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    demo_gradient_computation()
    demo_backprop_visualization()
    demo_neural_network()

    print("=" * 60)
    print("ALL DEMOS COMPLETE")
    print("=" * 60)
    print()
    print("Key takeaways:")
    print("1. Autograd tracks operations and computes gradients automatically")
    print("2. Backpropagation is just the chain rule applied recursively")
    print("3. Neural networks are compositions of simple differentiable operations")
    print("4. You now know how PyTorch works under the hood")
    print()


if __name__ == "__main__":
    main()
