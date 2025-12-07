"""
MicroGrad-NumPy: A Scalar-Value Autograd Engine
================================================

A minimal, educational implementation of reverse-mode automatic differentiation.
Built from scratch using only NumPy. No PyTorch, no TensorFlow, no magic.

The core insight: every operation in a neural network is just a composition of
simple mathematical operations. If we track those operations and know their
local derivatives, we can chain them together via the chain rule to compute
gradients of any output with respect to any input.

This is what PyTorch does under the hood. Now you know how to build PyTorch.

Author: Built as a senior-level portfolio project
"""

from __future__ import annotations
import math
import numpy as np
from typing import Union, Tuple, Set, List, Callable, Optional


# Type alias for numeric inputs
Numeric = Union[int, float, np.floating]


class Value:
    """
    A scalar value that tracks its computational history for automatic differentiation.
    
    Every Value knows:
    1. Its data (the actual number)
    2. Its gradient (derivative of the loss with respect to this value)
    3. Its parents (the Values that produced it via some operation)
    4. Its backward function (how to propagate gradients to its parents)
    
    The gradient is computed lazily via backward(). Until you call backward(),
    all gradients remain at 0.0.
    
    Attributes:
        data: The scalar value stored in this node.
        grad: The gradient of the final output with respect to this value.
        label: Optional name for debugging and visualization.
        
    Example:
        >>> a = Value(2.0, label='a')
        >>> b = Value(3.0, label='b')
        >>> c = a * b + a
        >>> c.backward()
        >>> print(a.grad)  # dc/da = b + 1 = 4.0
        4.0
        >>> print(b.grad)  # dc/db = a = 2.0
        2.0
    """
    
    __slots__ = ('data', 'grad', '_backward', '_prev', '_op', 'label')
    
    def __init__(
        self, 
        data: Numeric, 
        _children: Tuple[Value, ...] = (), 
        _op: str = '',
        label: str = ''
    ) -> None:
        """
        Initialize a Value node.
        
        Args:
            data: The scalar value to store.
            _children: Parent nodes in the computation graph (internal use).
            _op: The operation that produced this node (internal use).
            label: Optional name for debugging.
            
        Raises:
            TypeError: If data is not a numeric type.
        """
        if not isinstance(data, (int, float, np.floating)):
            raise TypeError(
                f"Value data must be numeric, got {type(data).__name__}"
            )
        
        self.data: float = float(data)
        self.grad: float = 0.0
        self._backward: Callable[[], None] = lambda: None
        self._prev: Set[Value] = set(_children)
        self._op: str = _op
        self.label: str = label
    
    def __repr__(self) -> str:
        """String representation showing data and gradient."""
        if self.label:
            return f"Value({self.label}={self.data:.4f}, grad={self.grad:.4f})"
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"
    
    # =========================================================================
    # Arithmetic Operations
    # =========================================================================
    
    def __add__(self, other: Union[Value, Numeric]) -> Value:
        """
        Addition: out = self + other
        
        Local derivatives:
            d(out)/d(self) = 1
            d(out)/d(other) = 1
            
        Args:
            other: Value or numeric to add.
            
        Returns:
            New Value representing the sum.
        """
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        
        def _backward() -> None:
            # Chain rule: gradient flows unchanged through addition
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        
        out._backward = _backward
        return out
    
    def __radd__(self, other: Numeric) -> Value:
        """Handle numeric + Value."""
        return self + other
    
    def __neg__(self) -> Value:
        """Negation: -self."""
        return self * -1
    
    def __sub__(self, other: Union[Value, Numeric]) -> Value:
        """Subtraction: self - other."""
        return self + (-other)
    
    def __rsub__(self, other: Numeric) -> Value:
        """Handle numeric - Value."""
        return other + (-self)
    
    def __mul__(self, other: Union[Value, Numeric]) -> Value:
        """
        Multiplication: out = self * other
        
        Local derivatives:
            d(out)/d(self) = other.data
            d(out)/d(other) = self.data
            
        Args:
            other: Value or numeric to multiply.
            
        Returns:
            New Value representing the product.
        """
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        
        def _backward() -> None:
            # Product rule derivatives
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        
        out._backward = _backward
        return out
    
    def __rmul__(self, other: Numeric) -> Value:
        """Handle numeric * Value."""
        return self * other
    
    def __truediv__(self, other: Union[Value, Numeric]) -> Value:
        """Division: self / other = self * other^(-1)."""
        return self * (other ** -1)
    
    def __rtruediv__(self, other: Numeric) -> Value:
        """Handle numeric / Value."""
        return other * (self ** -1)
    
    def __pow__(self, n: Union[int, float]) -> Value:
        """
        Power: out = self^n (where n is a constant, not a Value)
        
        Local derivative:
            d(out)/d(self) = n * self^(n-1)
            
        Args:
            n: The exponent (must be numeric, not Value).
            
        Returns:
            New Value representing self raised to power n.
            
        Raises:
            TypeError: If n is a Value (not supported).
        """
        if isinstance(n, Value):
            raise TypeError(
                "Power with Value exponent not supported. "
                "Use exp(n * log(self)) instead."
            )
        
        out = Value(self.data ** n, (self,), f'**{n}')
        
        def _backward() -> None:
            # Power rule: d/dx(x^n) = n * x^(n-1)
            self.grad += n * (self.data ** (n - 1)) * out.grad
        
        out._backward = _backward
        return out
    
    # =========================================================================
    # Activation Functions
    # =========================================================================
    
    def tanh(self) -> Value:
        """
        Hyperbolic tangent activation: out = tanh(self)
        
        tanh(x) = (e^(2x) - 1) / (e^(2x) + 1)
        
        Local derivative:
            d(tanh(x))/dx = 1 - tanh(x)^2
            
        Returns:
            New Value with tanh applied.
        """
        t = math.tanh(self.data)
        out = Value(t, (self,), 'tanh')
        
        def _backward() -> None:
            # tanh derivative: 1 - tanh^2
            self.grad += (1 - t ** 2) * out.grad
        
        out._backward = _backward
        return out
    
    def relu(self) -> Value:
        """
        Rectified Linear Unit: out = max(0, self)
        
        Local derivative:
            d(relu(x))/dx = 1 if x > 0 else 0
            
        Returns:
            New Value with ReLU applied.
        """
        out = Value(max(0, self.data), (self,), 'relu')
        
        def _backward() -> None:
            # ReLU derivative: pass gradient if input was positive
            self.grad += (self.data > 0) * out.grad
        
        out._backward = _backward
        return out
    
    def sigmoid(self) -> Value:
        """
        Sigmoid activation: out = 1 / (1 + e^(-self))
        
        Local derivative:
            d(sigmoid(x))/dx = sigmoid(x) * (1 - sigmoid(x))
            
        Returns:
            New Value with sigmoid applied.
        """
        s = 1 / (1 + math.exp(-self.data))
        out = Value(s, (self,), 'sigmoid')
        
        def _backward() -> None:
            # Sigmoid derivative: s * (1 - s)
            self.grad += s * (1 - s) * out.grad
        
        out._backward = _backward
        return out
    
    def exp(self) -> Value:
        """
        Exponential: out = e^self
        
        Local derivative:
            d(e^x)/dx = e^x
            
        Returns:
            New Value with exp applied.
        """
        e = math.exp(self.data)
        out = Value(e, (self,), 'exp')
        
        def _backward() -> None:
            # exp derivative: e^x
            self.grad += e * out.grad
        
        out._backward = _backward
        return out
    
    def log(self) -> Value:
        """
        Natural logarithm: out = ln(self)
        
        Local derivative:
            d(ln(x))/dx = 1/x
            
        Returns:
            New Value with log applied.
            
        Raises:
            ValueError: If self.data <= 0.
        """
        if self.data <= 0:
            raise ValueError(f"log undefined for non-positive values: {self.data}")
        
        out = Value(math.log(self.data), (self,), 'log')
        
        def _backward() -> None:
            # log derivative: 1/x
            self.grad += (1 / self.data) * out.grad
        
        out._backward = _backward
        return out
    
    # =========================================================================
    # Backpropagation
    # =========================================================================
    
    def backward(self) -> None:
        """
        Compute gradients for all nodes in the computation graph.
        
        This implements reverse-mode automatic differentiation (backpropagation).
        
        The algorithm:
        1. Build a topological ordering of the computation graph
        2. Set this node's gradient to 1.0 (d(self)/d(self) = 1)
        3. Walk backward through the graph, calling each node's _backward()
        
        After calling backward(), every Value in the graph will have its
        .grad attribute populated with the derivative of this Value with
        respect to that Value.
        
        Note: Calling backward() multiple times will ACCUMULATE gradients.
        Call zero_grad() first if you want fresh gradients.
        
        Example:
            >>> x = Value(2.0)
            >>> y = x ** 2 + 3 * x
            >>> y.backward()
            >>> print(x.grad)  # dy/dx = 2x + 3 = 7.0
            7.0
        """
        # Build topological order using DFS
        topo: List[Value] = []
        visited: Set[Value] = set()
        
        def build_topo(v: Value) -> None:
            if v not in visited:
                visited.add(v)
                for parent in v._prev:
                    build_topo(parent)
                topo.append(v)
        
        build_topo(self)
        
        # Seed gradient: d(self)/d(self) = 1
        self.grad = 1.0
        
        # Walk backward through the graph
        for node in reversed(topo):
            node._backward()
    
    def zero_grad(self) -> None:
        """
        Reset gradient to zero.
        
        Call this before a new backward pass if you don't want gradient
        accumulation.
        """
        self.grad = 0.0
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def item(self) -> float:
        """Return the scalar value (PyTorch compatibility)."""
        return self.data
    
    @staticmethod
    def zero_grad_all(values: List[Value]) -> None:
        """
        Zero gradients for a list of Values.
        
        Args:
            values: List of Value objects to zero.
        """
        for v in values:
            v.grad = 0.0


def topological_sort(root: Value) -> List[Value]:
    """
    Compute topological ordering of computation graph rooted at `root`.
    
    This is the order in which nodes must be processed during backpropagation.
    Nodes are ordered so that a node appears after all nodes that depend on it.
    
    Args:
        root: The root node of the computation graph.
        
    Returns:
        List of Values in topological order (root is last).
        
    Example:
        >>> a = Value(1.0)
        >>> b = Value(2.0)
        >>> c = a + b
        >>> d = c * a
        >>> topo = topological_sort(d)
        >>> # topo will be [a, b, c, d] or [b, a, c, d]
    """
    topo: List[Value] = []
    visited: Set[Value] = set()
    
    def dfs(v: Value) -> None:
        if v not in visited:
            visited.add(v)
            for parent in v._prev:
                dfs(parent)
            topo.append(v)
    
    dfs(root)
    return topo


def draw_graph(root: Value, format: str = 'text') -> str:
    """
    Generate a visualization of the computation graph.
    
    Args:
        root: Root node of the graph to visualize.
        format: 'text' for ASCII art, 'dot' for Graphviz DOT format.
        
    Returns:
        String representation of the graph.
    """
    nodes = topological_sort(root)
    node_ids = {n: i for i, n in enumerate(nodes)}
    
    if format == 'dot':
        lines = ['digraph G {', '  rankdir=LR;']
        for node in nodes:
            nid = node_ids[node]
            label = node.label if node.label else f'v{nid}'
            lines.append(
                f'  n{nid} [label="{label}\\n'
                f'data={node.data:.4f}\\n'
                f'grad={node.grad:.4f}", shape=box];'
            )
            if node._op:
                op_id = f'op{nid}'
                lines.append(f'  {op_id} [label="{node._op}", shape=circle];')
                lines.append(f'  {op_id} -> n{nid};')
                for parent in node._prev:
                    pid = node_ids[parent]
                    lines.append(f'  n{pid} -> {op_id};')
        lines.append('}')
        return '\n'.join(lines)
    
    else:  # text format
        lines = ['Computation Graph:', '=' * 50]
        for node in reversed(nodes):
            nid = node_ids[node]
            label = node.label if node.label else f'v{nid}'
            op_str = f' = {node._op}(' if node._op else ''
            if node._prev:
                parent_labels = [
                    nodes[node_ids[p]].label or f'v{node_ids[p]}' 
                    for p in node._prev
                ]
                op_str += ', '.join(parent_labels) + ')'
            lines.append(
                f'{label:>10}: data={node.data:>10.4f}, '
                f'grad={node.grad:>10.4f}{op_str}'
            )
        return '\n'.join(lines)
