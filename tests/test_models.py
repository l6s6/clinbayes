import itertools

import pytest
from clinbayes.bayes_engine.models import BayesianNetwork, CPT, Variable

DEFAULT_NAME: str = "pneumonia"
DEFAULT_STATES: tuple[str, ...] = ("no", "yes")


def _get_variable(
    name: str = DEFAULT_NAME,
    states: tuple[str, ...] = DEFAULT_STATES,
    parents: tuple[str, ...] = (),
) -> Variable:
    return Variable(name=name, states=states, parents=parents)


def _dummy_binary_cpt(var: Variable) -> CPT:
    keys = itertools.product(*(DEFAULT_STATES for _ in var.parents))
    return CPT(variable=var, table={key: (0.95, 0.05) for key in keys})


@pytest.fixture
def valid_prior() -> tuple[Variable, CPT]:
    var = _get_variable()
    cpt = CPT(variable=var, table={(): (0.95, 0.05)})
    return var, cpt


# --- Variable ---


def test_variable_name_missing_raises() -> None:
    with pytest.raises(ValueError, match="missing"):
        Variable(name="", states=DEFAULT_STATES, parents=())


def test_variable_self_parent_raises() -> None:
    with pytest.raises(ValueError, match="itself as parent"):
        Variable(name=DEFAULT_NAME, states=DEFAULT_STATES, parents=(DEFAULT_NAME,))


def test_variable_has_no_states_raises() -> None:
    with pytest.raises(ValueError, match="at least 1 state"):
        Variable(name=DEFAULT_NAME, states=(), parents=())


def test_variable_state_duplicate_raises() -> None:
    with pytest.raises(ValueError, match="duplicate states"):
        Variable(name=DEFAULT_NAME, states=("yes", "yes"), parents=())


def test_variable_parent_duplicate_raises() -> None:
    var = _get_variable()
    with pytest.raises(ValueError, match="duplicate parents"):
        Variable(name="a", states=DEFAULT_STATES, parents=(var.name, var.name))


# --- CPT ---


def test_cpt_negative_probability_raises() -> None:
    var = _get_variable()
    with pytest.raises(ValueError, match="probability must be in"):
        CPT(variable=var, table={(): (-0.1, 1.1)})


def test_cpt_sum_too_small_raises() -> None:
    var = _get_variable()
    with pytest.raises(ValueError, match="row sum is"):
        CPT(variable=var, table={(): (0.19, 0.8)})


def test_cpt_sum_too_big_raises() -> None:
    var = _get_variable()
    with pytest.raises(ValueError, match="row sum is"):
        CPT(variable=var, table={(): (0.21, 0.8)})


def test_cpt_table_missing_raises() -> None:
    var = _get_variable()
    with pytest.raises(ValueError, match="is empty"):
        CPT(variable=var, table={})


def test_cpt_table_invalid_col_count_raises() -> None:
    var = _get_variable()
    with pytest.raises(ValueError, match="parent key .* has length"):
        CPT(variable=var, table={("parent",): (0.21, 0.8)})


def test_cpt_table_invalid_row_count_raises() -> None:
    var = _get_variable()
    with pytest.raises(ValueError, match="row has invalid length"):
        CPT(variable=var, table={(): (0.21, 0.7, 0.1)})


# --- BayesianNetwork ---


def test_network_valid_single_node(valid_prior: tuple[Variable, CPT]) -> None:
    var, cpt = valid_prior
    net = BayesianNetwork(variables={var.name: var}, cpts={var.name: cpt})
    assert not net.has_cycle()


def test_network_3_nodes() -> None:
    var_a = _get_variable(name="a")
    var_b = _get_variable(name="b")
    var_c = _get_variable(name="c", parents=(var_a.name, var_b.name))
    net = BayesianNetwork(
        variables={var_a.name: var_a, var_b.name: var_b, var_c.name: var_c},
        cpts={
            var_a.name: _dummy_binary_cpt(var_a),
            var_b.name: _dummy_binary_cpt(var_b),
            var_c.name: _dummy_binary_cpt(var_c),
        },
    )
    assert not net.has_cycle()


def test_network_mismatch_keys_raises(valid_prior: tuple[Variable, CPT]) -> None:
    var, cpt = valid_prior
    with pytest.raises(ValueError, match="mismatched keys"):
        BayesianNetwork(variables={var.name: var}, cpts={"a": cpt})


def test_network_variable_name_mismatch_raises(valid_prior: tuple[Variable, CPT]) -> None:
    var, cpt = valid_prior
    with pytest.raises(ValueError, match="does not match variable.name"):
        BayesianNetwork(variables={"a": var}, cpts={"a": cpt})


def test_network_cpt_name_mismatch_raises() -> None:
    var_a = _get_variable(name="a")
    var_b = _get_variable(name="b")
    cpt_a = CPT(var_a, table={(): (0.95, 0.05)})
    cpt_b = CPT(var_b, table={(): (0.95, 0.05)})
    with pytest.raises(ValueError, match="references mismatched variable"):
        BayesianNetwork(
            variables={var_a.name: var_a, var_b.name: var_b},
            cpts={var_a.name: cpt_b, var_b.name: cpt_a},
        )


def test_cpt_parent_states_mismatch_raises() -> None:
    var_a = _get_variable(name="a")
    var_b = _get_variable(name="b", parents=(var_a.name,))
    cpt_a = CPT(var_a, table={(): (0.95, 0.05)})
    cpt_b = CPT(var_b, table={("x",): (0.95, 0.05), ("y",): (0.95, 0.05)})
    with pytest.raises(ValueError, match="does not match parent states"):
        BayesianNetwork(
            variables={var_a.name: var_a, var_b.name: var_b},
            cpts={var_a.name: cpt_a, var_b.name: cpt_b},
        )


def test_network_variable_unknown_parent_raises() -> None:
    var = _get_variable(parents=("a",))
    cpt = CPT(variable=var, table={("yes",): (0.3, 0.7), ("no",): (0.85, 0.15)})
    with pytest.raises(ValueError, match="references unknown parent"):
        BayesianNetwork(variables={var.name: var}, cpts={var.name: cpt})


def test_network_cycle_raises() -> None:
    var_a = _get_variable(name="a", parents=("b",))
    var_b = _get_variable(name="b", parents=(var_a.name,))
    with pytest.raises(ValueError, match="contains a cycle"):
        BayesianNetwork(
            variables={var_a.name: var_a, var_b.name: var_b},
            cpts={var_a.name: _dummy_binary_cpt(var_a), var_b.name: _dummy_binary_cpt(var_b)},
        )


def test_network_big_cycle_raises() -> None:
    var_a = _get_variable(name="a", parents=("c",))
    var_b = _get_variable(name="b", parents=(var_a.name,))
    var_c = _get_variable(name="c", parents=(var_b.name,))
    with pytest.raises(ValueError, match="contains a cycle"):
        BayesianNetwork(
            variables={var_a.name: var_a, var_b.name: var_b, var_c.name: var_c},
            cpts={
                var_a.name: _dummy_binary_cpt(var_a),
                var_b.name: _dummy_binary_cpt(var_b),
                var_c.name: _dummy_binary_cpt(var_c),
            },
        )

