import jax.numpy as jnp
import networkx as nx

class RegularTypeSystem:
    def __init__(self):
        self.variables = {}

    def declare(self, var_name, value):
        self.variables[var_name] = value

    def get(self, var_name):
        return self.variables.get(var_name, None)


class DependentTypeSystem:
    def __init__(self):
        self.variables = {}

    def declare(self, var_name, var_type, value):
        if isinstance(value, var_type):
            self.variables[var_name] = value
        else:
            raise TypeError(f"Expected type {var_type}, got {type(value)}")

    def get(self, var_name):
        return self.variables.get(var_name, None)


class IndependentTypeSystem:
    def __init__(self):
        self.variables = {}

    def declare(self, var_name, value):
        self.variables[var_name] = value

    def get(self, var_name):
        return self.variables.get(var_name, None)


class QueryBasedTypeSystem:
    def __init__(self):
        self.variables = {}

    def declare(self, var_name, value):
        self.variables[var_name] = value

    def query(self, var_name, dependency_name):
        if dependency_name in self.variables:
            self.variables[var_name] = self.variables[dependency_name]
        else:
            raise ValueError(f"Dependency {dependency_name} not found")

    def get(self, var_name):
        return self.variables.get(var_name, None)


class VariableTypes:
    INT = int
    STR = str
    FLOAT = float
    DOUBLE = float  # In Python, float is double precision
    LIST = list
    DICT = dict
    NARRAY = jnp.ndarray
    TENSOR = jnp.ndarray
    GRAP = nx.Graph
    TUPLE = tuple
    BOOL = bool
    COMPLEX = complex
    SET = set