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
            raise ValueError(f"Variable '{self.name}' cannot have itself as parent")
        if len(self.states) < 1:
            raise ValueError(f"Variable '{self.name}' must have at least 1 state")
        if len(set(self.states)) != len(self.states):
            raise ValueError(f"Variable '{self.name}' has duplicate states: {self.states}")
        if len(set(self.parents)) != len(self.parents):
            raise ValueError(f"Variable '{self.name}' has duplicate parents: {self.parents}")


@dataclass(frozen=True)
class CPT:
    variable: Variable
    table: dict[tuple[str, ...], tuple[float, ...]]

    def __post_init__(self) -> None:
        if not self.table:
            raise ValueError(f"CPT for '{self.variable.name}' is empty")

        for parent_values, row in self.table.items():
            if len(row) != len(self.variable.states):
                raise ValueError(
                    f"CPT for '{self.variable.name}' at key {parent_values}: "
                    f"row has invalid length {len(row)}, expected {len(self.variable.states)}"
                )

            if len(parent_values) != len(self.variable.parents):
                raise ValueError(
                    f"CPT for '{self.variable.name}': "
                    f"parent key {parent_values} has length {len(parent_values)}, expected {len(self.variable.parents)}"
                )

            for item in row:
                if not (0.0 <= item <= 1.0):
                    raise ValueError(
                        f"CPT for '{self.variable.name}' at key {parent_values}: "
                        f"probability must be in [0, 1], got {item}"
                    )

            if not math.isclose(sum(row), 1.0, abs_tol=1e-6):
                raise ValueError(
                    f"CPT for '{self.variable.name}' at key {parent_values}: "
                    f"row sum is {sum(row)}, expected 1.0"
                )


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
            diff = sorted(set(self.variables.keys()) ^ set(self.cpts.keys()))
            raise ValueError(f"Variable set does not match CPT variable set; mismatched keys: {diff}")

        for name, variable in self.variables.items():
            if name != variable.name:
                raise ValueError(f"Variable dict key '{name}' does not match variable.name '{variable.name}'")

        for variable in self.variables.values():
            for parent in variable.parents:
                if parent not in self.variables:
                    raise ValueError(f"Variable '{variable.name}' references unknown parent '{parent}'")

        for name, cpt in self.cpts.items():
            if cpt.variable != self.variables[name]:
                raise ValueError(
                    f"CPT for '{name}' references mismatched variable '{cpt.variable.name}'"
                )

            parents = cpt.variable.parents
            expected_keys = set(itertools.product(*(self.variables[p].states for p in parents)))
            actual_keys = set(cpt.table.keys())
            if actual_keys != expected_keys:
                missing = sorted(expected_keys - actual_keys)
                extra = sorted(actual_keys - expected_keys)
                raise ValueError(
                    f"CPT for '{name}' does not match parent states (missing={missing}, extra={extra})"
                )

        if self.has_cycle():
            raise ValueError("Graph contains a cycle")