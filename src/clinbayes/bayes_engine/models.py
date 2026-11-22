from dataclasses import dataclass

import itertools

import math


@dataclass(frozen=True)
class Variable:
    name: str
    states: tuple[str, ...]
    parents: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Variable name is missing")
        if self.name in self.parents:
            raise ValueError("Variable cannot have itself as parent")
        if len(self.states) < 1:
            raise ValueError("Variable must have at least 1 state")
        if len(set(self.states)) != len(self.states):
            raise ValueError("Duplicate in variable states")
        if len(set(self.parents)) != len(self.parents):
            raise ValueError("Duplicate in variable parents")


@dataclass(frozen=True)
class CPT:
    variable: Variable
    table: dict[tuple[str, ...], tuple[float, ...]]

    def __post_init__(self) -> None:
        if not self.table:
            raise ValueError("Table is missing")

        for parent_values, row in self.table.items():
            if len(row) != len(self.variable.states):
                raise ValueError(f"Row with invalid length {len(row)}, expected {len(self.variable.states)}")

            if len(parent_values) != len(self.variable.parents):
                raise ValueError(f"Column with invalid length {len(parent_values)}, expected {len(self.variable.parents)}")

            for item in row:
                if not (0.0 <= item <= 1.0):
                    raise ValueError(f"Probability must be in [0, 1], got {item}")

            if not math.isclose(sum(row), 1.0, abs_tol=1e-6):
                raise ValueError(f"Row sum is {sum(row)}, expected 1.0")


@dataclass(frozen=True)
class BayesianNetwork:
    variables: dict[str, Variable]
    cpts: dict[str, CPT]

    def _dfs_util(self, u, color) -> bool:
        gray = 1
        black = 2
        color[u] = gray

        for parent in self.variables[u].parents:
            if color[parent] == gray:  # Back edge found
                return True
            if color[parent] == 0 and self._dfs_util(parent, color):  # Visit unvisited node
                return True

        color[u] = black
        return False

    def has_cycle(self) -> bool:
        color: dict[str, int] = {name: 0 for name in self.variables}

        for name in self.variables:
            if color[name] == 0:
                if self._dfs_util(name, color):
                    return True
        return False

    def __post_init__(self) -> None:
        if set(self.variables.keys()) != set(self.cpts.keys()):
            raise ValueError(f"Variable set does not match CPT variable set")

        for name, variable in self.variables.items():
            if name != variable.name:
                raise ValueError(f"Variable name does not match its dict name: {variable.name} != {name}")

        for variable in self.variables.values():
            for parent in variable.parents:
                if parent not in self.variables:
                    raise ValueError(f"Variable has parent that is not a variable")

        for name, cpt in self.cpts.items():
            if cpt.variable != self.variables[name]:
                raise ValueError(f"CPT existent that does not belong to a variable")

            parents = cpt.variable.parents
            if set(cpt.table.keys()) != set(itertools.product(*(self.variables[p].states for p in parents))):
                raise ValueError(f"CPT tables columns dont match parent states")

        if self.has_cycle():
            raise ValueError("Graph contains a cycle")