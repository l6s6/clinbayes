from pathlib import Path
from pprint import pprint

import yaml

from clinbayes.bayes_engine.models import BayesianNetwork, Variable, CPT


def load_network(path: Path) -> BayesianNetwork:
    with path.open("r", encoding="utf-8") as f:
        raw_data = yaml.safe_load(f)

    if not isinstance(raw_data, dict) or not isinstance(raw_data.get("nodes"), list):
        raise ValueError(f"Error parsing YAML file: {path.name}")

    variables: dict[str, Variable] = {}
    cpts: dict[str, CPT] = {}

    for entry in raw_data['nodes']:
        variable = parse_variable(entry)
        if variable.name in variables:
            raise ValueError(f"Variable duplicate found: \"{variable.name}\"")
        variables[variable.name] = variable
        cpts[variable.name] = parse_cpt(entry, variable)

    return BayesianNetwork(variables=variables, cpts=cpts)


def parse_variable(entry) -> Variable:
    if not isinstance(entry, dict) or not {"name", "parents", "states"}.issubset(entry):
        raise ValueError("YAML contains corrupt variable")

    name = entry["name"]
    parents = entry["parents"]
    states = entry["states"]
    if not isinstance(name, str) or not isinstance(parents, list) or not isinstance(states, list):
        raise ValueError("Invalid types in variable definition")
    return Variable(name=name, parents=tuple(parents), states=tuple(states))


def parse_cpt(entry, variable) -> CPT:
    cpt_raw = entry.get("cpt")
    if not isinstance(cpt_raw, dict):
        raise ValueError("YAML variable is missing valid cpt")

    cpt_table: dict[tuple[str, ...], tuple[float, ...]] = {}
    for condition, probs in cpt_raw.items():
        if not isinstance(condition, str):
            raise ValueError(f"CPT key must be a quoted string, got {type(condition).__name__}")
        if not isinstance(probs, list):
            raise ValueError("CPT probabilities must be a list")

        elements_tuple = () if not condition.strip() else tuple(k.strip() for k in condition.split(","))
        cpt_table[elements_tuple] = tuple(float(p) for p in probs)

    return CPT(variable=variable, table=cpt_table)


if __name__ == "__main__":
    pprint(load_network(Path("../../../data/networks/mini_test.yaml")))