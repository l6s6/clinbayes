import textwrap
from pathlib import Path

import pytest

from clinbayes.bayes_engine.loader import load_network

DEFAULT_YAML_CONTENT = """
nodes:
  - name: A
    states: ["yes", "no"]
    parents: []
    cpt:
      "": [0.3, 0.7]
"""


def _write_yaml(tmp_path: Path, content: str) -> Path:
    yaml_file = tmp_path / "net.yaml"
    yaml_file.write_text(textwrap.dedent(content).lstrip(), encoding="utf-8")
    return yaml_file


# --- Load Network ---


def test_load_network_example(tmp_path: Path) -> None:
    yaml_file = _write_yaml(tmp_path, DEFAULT_YAML_CONTENT)
    net = load_network(yaml_file)
    assert "A" in net.variables


def test_load_empty_yaml_raises(tmp_path: Path) -> None:
    yaml_file = _write_yaml(tmp_path, "")
    with pytest.raises(ValueError, match="Invalid network YAML in"):
        load_network(yaml_file)


def test_load_duplicate_variable_raises(tmp_path: Path) -> None:
    yaml_file = _write_yaml(
        tmp_path,
        """
        nodes:
          - name: A
            states: ["yes", "no"]
            parents: []
            cpt:
              "": [0.3, 0.7]
          - name: A
            states: ["yes", "no"]
            parents: []
            cpt:
              "": [0.3, 0.7]
        """,
    )
    with pytest.raises(ValueError, match="Variable duplicate found"):
        load_network(yaml_file)


# --- Parse Variable ---


def test_parse_invalid_entry_node_raises(tmp_path: Path) -> None:
    yaml_file = _write_yaml(
        tmp_path,
        """
        nodes:
          - "not_a_mapping"
        """,
    )
    with pytest.raises(ValueError, match="Invalid node entry in YAML"):
        load_network(yaml_file)


def test_parse_variable_missing_field_raises(tmp_path: Path) -> None:
    yaml_file = _write_yaml(
        tmp_path,
        """
        nodes:
          - name: A
            parents: []
            cpt:
              "": [0.3, 0.7]
        """,
    )
    with pytest.raises(ValueError, match="is missing required fields"):
        load_network(yaml_file)


def test_parse_variable_missing_name_raises(tmp_path: Path) -> None:
    yaml_file = _write_yaml(
        tmp_path,
        """
        nodes:
          - name:
            states: ["yes", "no"]
            parents: []
            cpt:
              "": [0.3, 0.7]
        """,
    )
    with pytest.raises(ValueError, match="has invalid field types"):
        load_network(yaml_file)


# --- Parse CPT ---


def test_parse_cpt_missing_raises(tmp_path: Path) -> None:
    yaml_file = _write_yaml(
        tmp_path,
        """
        nodes:
          - name: A
            states: ["yes", "no"]
            parents: []
        """,
    )
    with pytest.raises(ValueError, match="is missing a valid 'cpt' mapping"):
        load_network(yaml_file)


def test_parse_cpt_invalid_condition_raises(tmp_path: Path) -> None:
    yaml_file = _write_yaml(
        tmp_path,
        """
        nodes:
          - name: A
            states: ["yes", "no"]
            parents: []
            cpt:
              "": [0.3, 0.7]
          - name: B
            states: ["yes", "no"]
            parents: [A]
            cpt:
              1: [0.3, 0.7]
              2: [0.3, 0.7]
        """,
    )
    with pytest.raises(ValueError, match="key must be a quoted string"):
        load_network(yaml_file)


def test_parse_cpt_probs_no_list_raises(tmp_path: Path) -> None:
    yaml_file = _write_yaml(
        tmp_path,
        """
        nodes:
          - name: A
            states: ["yes", "no"]
            parents: []
            cpt:
              "": 1
        """,
    )
    with pytest.raises(ValueError, match="probabilities must be a list"):
        load_network(yaml_file)


def test_parse_cpt_invalid_prob_raises(tmp_path: Path) -> None:
    yaml_file = _write_yaml(
        tmp_path,
        """
        nodes:
          - name: A
            states: ["yes", "no"]
            parents: []
            cpt:
              "": [0.1, A]
        """,
    )
    with pytest.raises(ValueError, match="invalid probability value in"):
        load_network(yaml_file)


# --- Integration ---


def test_load_mini_test_network() -> None:
    mini_test_path = Path(__file__).resolve().parents[1] / "data" / "networks" / "mini_test.yaml"
    net = load_network(mini_test_path)

    assert set(net.variables.keys()) == {"smoking", "pneumonia", "bronchitis", "crp", "dyspnea"}
    assert not net.has_cycle()
    assert net.variables["crp"].states == ("normal", "mildly_elevated", "strongly_elevated")
    assert net.variables["dyspnea"].parents == ("pneumonia", "bronchitis")
    assert net.cpts["dyspnea"].table[("yes", "no")] == (0.30, 0.70)

