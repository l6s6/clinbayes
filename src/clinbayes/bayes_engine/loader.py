from pathlib import Path
from pprint import pprint

import yaml

from clinbayes.bayes_engine.models import BayesianNetwork, Variable, CPT


def load_network(path: Path) -> BayesianNetwork:
    with path.open("r", encoding="utf-8") as f:
        raw_data = yaml.safe_load(f)

    if not isinstance(raw_data, dict) or not isinstance(raw_data.get("nodes"), list):
        raise ValueError(f"Invalid network YAML in '{path.name}': expected top-level mapping with 'nodes' list")

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
    if not isinstance(entry, dict):
        raise ValueError(f"Invalid node entry in YAML: expected mapping, got {type(entry).__name__}")
    missing = {"name", "parents", "states"} - set(entry.keys())
    if missing:
        var_label = entry.get("name", "<unknown>")
        raise ValueError(f"Variable '{var_label}' in YAML is missing required fields: {sorted(missing)}")

    name = entry["name"]
    parents = entry["parents"]
    states = entry["states"]
    if not isinstance(name, str) or not isinstance(parents, list) or not isinstance(states, list):
        raise ValueError(
            f"Variable '{name}' has invalid field types (expected str name, list parents, list states)"
        )
    return Variable(name=name, parents=tuple(parents), states=tuple(states))


def parse_cpt(entry, variable) -> CPT:
    cpt_raw = entry.get("cpt")
    if not isinstance(cpt_raw, dict):
        raise ValueError(f"Variable '{variable.name}' in YAML is missing a valid 'cpt' mapping")

    cpt_table: dict[tuple[str, ...], tuple[float, ...]] = {}
    for condition, probs in cpt_raw.items():
        if not isinstance(condition, str):
            raise ValueError(
                f"CPT for '{variable.name}': key must be a quoted string, got {type(condition).__name__} ({condition!r})"
            )
        if not isinstance(probs, list):
            raise ValueError(f"CPT for '{variable.name}' at key '{condition}': probabilities must be a list")

        elements_tuple = () if not condition.strip() else tuple(k.strip() for k in condition.split(","))
        try:
            cpt_table[elements_tuple] = tuple(float(p) for p in probs)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"CPT for '{variable.name}' at key '{condition}': invalid probability value in {probs!r}"
            ) from exc

    return CPT(variable=variable, table=cpt_table)


if __name__ == "__main__":
    pprint(load_network(Path("../../../data/networks/mini_test.yaml")))