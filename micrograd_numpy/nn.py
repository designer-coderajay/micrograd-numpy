"""
Neural Network Module
=====================

PyTorch-like neural network building blocks using our autograd engine.

This module provides:
- Module: Base class for all neural network components
- Neuron: A single neuron with weights, bias, and activation
- Layer: A collection of neurons (fully connected layer)
- MLP: Multi-layer perceptron (stack of layers)

The API mirrors PyTorch's nn.Module:
- model.parameters() returns all trainable parameters
- model.zero_grad() resets all gradients
- Forward pass is just calling the model: output = model(input)
"""

from __future__ import annotations
import random
import numpy as np
from typing import List, Union, Optional, Iterator
from .engine import Value  # import Value as value to avoid name conflict
class Module:
    """
    Base class for all neural network modules.

    Your models should subclass this class. Provides:
    - parameters(): collect all trainable Value objects
    - zero_grad(): reset gradients before backward pass

    This is the foundation of the PyTorch-like API.
    """

    def parameters(self) -> List[Value]:
        """
        Return all trainable parameters in this module.

        Override this in subclasses to return the module's parameters.

        Returns:
            List of Value objects representing trainable parameters.
        """
        return []

    def zero_grad(self) -> None:
        """
        Reset gradients of all parameters to zero.

        Call this before each backward pass to prevent gradient accumulation.
        """
        for p in self.parameters():
            p.grad = 0.0

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class Neuron(Module):
    """
    A single artificial neuron.

    Computes: output = activation(sum(w_i * x_i) + b)

    The neuron is the fundamental unit of neural networks. It:
    1. Takes multiple inputs
    2. Multiplies each by a learnable weight
    3. Adds them together with a learnable bias
    4. Applies a nonlinear activation function

    Attributes:
        w: List of weight Values
        b: Bias Value
        nonlin: Whether to apply nonlinearity
        activation: Which activation function to use

    Example:
        >>> n = Neuron(3, activation='relu')  # 3 inputs
        >>> x = [Value(1.0), Value(2.0), Value(3.0)]
        >>> out = n(x)  # Forward pass
    """

    def __init__(
        self,
        nin: int,
        nonlin: bool = True,
        activation: str = 'relu'
    ) -> None:
        """
        Initialize a neuron.

        Args:
            nin: Number of inputs to this neuron.
            nonlin: Whether to apply nonlinear activation.
            activation: Activation function ('relu', 'tanh', 'sigmoid').
        """
        # Xavier/Glorot initialization: helps with training stability
        scale = (2.0 / nin) ** 0.5
        self.w: List[Value] = [
            Value(random.uniform(-1, 1) * scale, label=f'w{i}')
            for i in range(nin)
        ]
        self.b: Value = Value(0.0, label='b')
        self.nonlin: bool = nonlin
        self.activation: str = activation

    def __call__(self, x: List[Union[Value, float]]) -> Value:
        """
        Forward pass: compute neuron output.

        Args:
            x: List of inputs (Values or floats).

        Returns:
            Single Value representing neuron output.

        Raises:
            ValueError: If input length doesn't match weight count.
        """
        if len(x) != len(self.w):
            raise ValueError(
                f"Expected {len(self.w)} inputs, got {len(x)}"
            )

        # Weighted sum: sum(w_i * x_i) + b
        act = sum(
            (wi * xi for wi, xi in zip(self.w, x)),
            start=self.b
        )

        # Apply activation
        if self.nonlin:
            if self.activation == 'relu':
                return act.relu()
            elif self.activation == 'tanh':
                return act.tanh()
            elif self.activation == 'sigmoid':
                return act.sigmoid()

        return act

    def parameters(self) -> List[Value]:
        """Return weights and bias."""
        return self.w + [self.b]

    def __repr__(self) -> str:
        act = self.activation if self.nonlin else 'Linear'
        return f"Neuron({len(self.w)}, {act})"


class Layer(Module):
    """
    A fully connected layer of neurons.

    A layer is just a collection of neurons that all receive the same input.
    Each neuron produces one output, so a layer with `nout` neurons
    transforms an input of size `nin` to an output of size `nout`.

    Attributes:
        neurons: List of Neuron objects

    Example:
        >>> layer = Layer(3, 4)  # 3 inputs, 4 outputs
        >>> x = [Value(1.0), Value(2.0), Value(3.0)]
        >>> out = layer(x)  # Returns list of 4 Values
    """

    def __init__(
        self,
        nin: int,
        nout: int,
        nonlin: bool = True,
        activation: str = 'relu'
    ) -> None:
        """
        Initialize a layer.

        Args:
            nin: Number of inputs per neuron.
            nout: Number of neurons (outputs).
            nonlin: Whether neurons use nonlinearity.
            activation: Activation function for all neurons.
        """
        self.neurons: List[Neuron] = [
            Neuron(nin, nonlin=nonlin, activation=activation)
            for _ in range(nout)
        ]

    def __call__(self, x: List[Union[Value, float]]) -> List[Value]:
        """
        Forward pass: compute all neuron outputs.

        Args:
            x: Input values.

        Returns:
            List of Values, one per neuron.
        """
        return [n(x) for n in self.neurons]

    def parameters(self) -> List[Value]:
        """Return all parameters from all neurons."""
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self) -> str:
        return f"Layer({len(self.neurons[0].w)} -> {len(self.neurons)})"


class MLP(Module):
    """
    Multi-Layer Perceptron: a stack of fully connected layers.

    This is the classic feedforward neural network. Data flows from input
    through each hidden layer to the output, with nonlinear activations
    between layers.

    Architecture:
        Input -> Hidden1 -> ... -> HiddenN -> Output

    All hidden layers use the specified activation. The output layer
    is linear (no activation) by default, which is standard for regression
    and classification (softmax/sigmoid applied separately).

    Attributes:
        layers: List of Layer objects

    Example:
        >>> # Create MLP: 3 inputs -> 4 hidden -> 4 hidden -> 1 output
        >>> model = MLP(3, [4, 4, 1])
        >>> x = [Value(1.0), Value(2.0), Value(3.0)]
        >>> out = model(x)  # Single output Value
    """

    def __init__(
        self,
        nin: int,
        nouts: List[int],
        activation: str = 'relu'
    ) -> None:
        """
        Initialize an MLP.

        Args:
            nin: Number of input features.
            nouts: List of layer sizes. Last element is output size.
            activation: Activation function for hidden layers.

        Example:
            MLP(3, [4, 4, 1]) creates:
            - Layer 1: 3 -> 4 (with activation)
            - Layer 2: 4 -> 4 (with activation)
            - Layer 3: 4 -> 1 (linear output)
        """
        sizes = [nin] + nouts
        self.layers: List[Layer] = []

        for i in range(len(nouts)):
            # Last layer is linear (no activation)
            is_output = (i == len(nouts) - 1)
            self.layers.append(
                Layer(
                    sizes[i],
                    sizes[i + 1],
                    nonlin=not is_output,
                    activation=activation
                )
            )

    def __call__(self, x: List[Union[Value, float]]) -> Union[Value, List[Value]]:
        """
        Forward pass through all layers.

        Args:
            x: Input values.

        Returns:
            Output Value(s). Returns single Value if output size is 1,
            otherwise returns list of Values.
        """
        for layer in self.layers:
            x = layer(x)

        # Unwrap single-element output
        return x[0] if len(x) == 1 else x

    def parameters(self) -> List[Value]:
        """Return all parameters from all layers."""
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self) -> str:
        layer_strs = [str(layer) for layer in self.layers]
        return f"MLP([{', '.join(layer_strs)}])"


# =============================================================================
# Loss Functions
# =============================================================================

def mse_loss(predictions: List[Value], targets: List[float]) -> Value:
    """
    Mean Squared Error loss.

    MSE = (1/n) * sum((pred_i - target_i)^2)

    Standard loss for regression problems.

    Args:
        predictions: Model outputs (Values).
        targets: Ground truth values (floats).

    Returns:
        Scalar Value representing the loss.
    """
    n = len(predictions)
    return sum(
        (pred - target) ** 2
        for pred, target in zip(predictions, targets)
    ) / n


def binary_cross_entropy(
    predictions: List[Value],
    targets: List[float],
    eps: float = 1e-7
) -> Value:
    """
    Binary Cross-Entropy loss.

    BCE = -(1/n) * sum(y*log(p) + (1-y)*log(1-p))

    Standard loss for binary classification. Predictions should be
    probabilities (0-1), typically from a sigmoid output.

    Args:
        predictions: Model outputs (Values in range 0-1).
        targets: Binary labels (0 or 1).
        eps: Small constant for numerical stability.

    Returns:
        Scalar Value representing the loss.
    """
    n = len(predictions)
    total = Value(0.0)

    for pred, target in zip(predictions, targets):
        # Clamp predictions for numerical stability
        p = pred.data
        p = max(eps, min(1 - eps, p))
        p_val = Value(p)

        # BCE formula
        if target == 1:
            total = total + (-p_val.log())
        else:
            total = total + (-(1 - p_val).log())

    return total / n


def hinge_loss(predictions: List[Value], targets: List[float]) -> Value:
    """
    Hinge loss for SVM-style classification.

    Hinge = (1/n) * sum(max(0, 1 - y * pred))

    Targets should be -1 or +1.

    Args:
        predictions: Model outputs (Values).
        targets: Labels (-1 or +1).

    Returns:
        Scalar Value representing the loss.
    """
    n = len(predictions)
    total = Value(0.0)

    for pred, target in zip(predictions, targets):
        margin = 1 - target * pred
        total = total + margin.relu()

    return total / n


# =============================================================================
# Optimizers
# =============================================================================

class SGD:
    """
    Stochastic Gradient Descent optimizer.

    Updates parameters: p = p - lr * p.grad

    This is the simplest optimizer. It just moves parameters in the
    direction opposite to the gradient, scaled by the learning rate.

    Attributes:
        params: List of parameters to optimize.
        lr: Learning rate.
    """

    def __init__(self, params: List[Value], lr: float = 0.01) -> None:
        """
        Initialize SGD optimizer.

        Args:
            params: Parameters to optimize.
            lr: Learning rate (step size).
        """
        self.params = params
        self.lr = lr

    def step(self) -> None:
        """
        Perform one optimization step.

        Updates all parameters based on their gradients.
        Call this after backward().
        """
        for p in self.params:
            p.data -= self.lr * p.grad

    def zero_grad(self) -> None:
        """Reset all gradients to zero."""
        for p in self.params:
            p.grad = 0.0


class Adam:
    """
    Adam optimizer: Adaptive Moment Estimation.

    Adam combines momentum (first moment) with RMSprop (second moment)
    for adaptive per-parameter learning rates. Generally works better
    than vanilla SGD for most problems.

    Update rules:
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad^2
        m_hat = m / (1 - beta1^t)
        v_hat = v / (1 - beta2^t)
        p = p - lr * m_hat / (sqrt(v_hat) + eps)

    Attributes:
        params: Parameters to optimize.
        lr: Learning rate.
        beta1: Exponential decay rate for first moment.
        beta2: Exponential decay rate for second moment.
        eps: Small constant for numerical stability.
    """

    def __init__(
        self,
        params: List[Value],
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8
    ) -> None:
        """
        Initialize Adam optimizer.

        Args:
            params: Parameters to optimize.
            lr: Learning rate.
            beta1: First moment decay (default 0.9).
            beta2: Second moment decay (default 0.999).
            eps: Numerical stability constant.
        """
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        # Initialize moment estimates
        self.m: List[float] = [0.0] * len(params)  # First moment
        self.v: List[float] = [0.0] * len(params)  # Second moment
        self.t: int = 0  # Time step

    def step(self) -> None:
        """
        Perform one Adam optimization step.

        Updates all parameters using adaptive learning rates.
        """
        self.t += 1

        for i, p in enumerate(self.params):
            g = p.grad

            # Update biased first moment estimate
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g

            # Update biased second moment estimate
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g ** 2)

            # Bias correction
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            # Update parameter
            p.data -= self.lr * m_hat / (v_hat ** 0.5 + self.eps)

    def zero_grad(self) -> None:
        """Reset all gradients to zero."""
        for p in self.params:
            p.grad = 0.0
