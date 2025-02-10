import jax.numpy as jnp
import torch
import networkx as nx
from typing import Any

class BaseType:
    """
    Base class for all types in the language.
    """
    def __init__(self, value: Any):
        self.value = value
    
    def __repr__(self):
        return f"{self.__class__.__name__}({self.value})"

class IntType(BaseType):
    def __init__(self, value: int):
        if not isinstance(value, int):
            raise TypeError("Expected an integer")
        super().__init__(value)

class FloatType(BaseType):
    def __init__(self, value: float):
        if not isinstance(value, float):
            raise TypeError("Expected a float")
        super().__init__(value)

class DoubleType(BaseType):
    def __init__(self, value: float):
        if not isinstance(value, float):
            raise TypeError("Expected a double (float)")
        super().__init__(value)

class StringType(BaseType):
    def __init__(self, value: str):
        if not isinstance(value, str):
            raise TypeError("Expected a string")
        super().__init__(value)

class TupleType(BaseType):
    def __init__(self, value: tuple):
        if not isinstance(value, tuple):
            raise TypeError("Expected a tuple")
        super().__init__(value)

class ListType(BaseType):
    def __init__(self, value: list):
        if not isinstance(value, list):
            raise TypeError("Expected a list")
        super().__init__(value)

class DictType(BaseType):
    def __init__(self, value: dict):
        if not isinstance(value, dict):
            raise TypeError("Expected a dictionary")
        super().__init__(value)

class SetType(BaseType):
    def __init__(self, value: set):
        if not isinstance(value, set):
            raise TypeError("Expected a set")
        super().__init__(value)

class BoolType(BaseType):
    def __init__(self, value: bool):
        if not isinstance(value, bool):
            raise TypeError("Expected a boolean")
        super().__init__(value)

class NdArrayType(BaseType):
    def __init__(self, value):
        if not isinstance(value, jnp.ndarray):
            raise TypeError("Expected a JAX ndarray")
        super().__init__(value)

class TensorType(BaseType):
    def __init__(self, value):
        if not isinstance(value, torch.Tensor):
            raise TypeError("Expected a PyTorch tensor")
        super().__init__(value)

class BytesType(BaseType):
    def __init__(self, value: bytes):
        if not isinstance(value, bytes):
            raise TypeError("Expected a bytes object")
        super().__init__(value)

class GraphType(BaseType):
    def __init__(self, value):
        if not isinstance(value, nx.Graph):
            raise TypeError("Expected a NetworkX Graph")
        super().__init__(value)
