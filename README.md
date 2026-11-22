# clinbayes

A lightweight Python engine and planned full-stack application for clinical Bayesian networks—spanning a custom inference engine, a minimal FHIR parser, a FastAPI backend, and a React/TypeScript UI.

## Current Status

The foundation of **`bayes_engine`** (Milestone 1) is implemented:

- **Type-safe data models** (`Variable`, `CPT`, `BayesianNetwork`) built with immutable dataclasses (`@dataclass(frozen=True)`)
- **Arbitrary finite discrete states** (not limited to binary variables, e.g., `normal`, `mildly_elevated`, `strongly_elevated`)
- **Structural & mathematical validation**:
  - Probabilities in `[0, 1]` and row sums equal to `1.0` (`math.isclose(..., abs_tol=1e-6)`)
  - Completeness check of CPT keys across all parent state combinations (`itertools.product`)
  - Cycle detection in the directed graph via 3-color DFS in $O(|V| + |E|)$
- **YAML loader** (`load_network`) to parse network definitions
- **Sample network** at `data/networks/mini_test.yaml` (`smoking`, `pneumonia`, `bronchitis`, `crp`, `dyspnea`)
- **Unit tests** using `pytest` for valid and invalid networks

## Roadmap (Core-First to v1.0)

Development follows an incremental **core-first** approach:

1. **Complete `bayes_engine`**
   - Factor algebra (`Factor` objects)
   - Exact inference via **Variable Elimination** (verified against brute-force enumeration / `pgmpy` as a test oracle in `pytest`)
   - **Explainability / sensitivity analysis:** Leave-One-Out impact of each observed evidence item ($P(D \mid E) - P(D \mid E \setminus \{e_i\})$)
2. **`fhir_parser` (minimal integration)**
   - Mapping clinical observations to discrete network states for 5–6 fixed LOINC codes using Pydantic
3. **`api` (backend)**
   - FastAPI service exposing `/patients`, `/graph`, and `/infer`
4. **Frontend**
   - Lean React/TypeScript UI for graph visualization, evidence input, and explainability output

## Project Structure

```text
clinbayes/
├── data/
│   └── networks/
│       └── mini_test.yaml
├── src/
│   └── clinbayes/
│       └── bayes_engine/
│           ├── loader.py
│           └── models.py
├── tests/
│   ├── test_loader.py
│   └── test_models.py
└── pyproject.toml
```

## Requirements

- Python `>= 3.11`
- `pyyaml >= 6.0`

## Installation

```bash
pip install -e .
```

For development and testing:

```bash
pip install -e ".[dev]"
```

## Usage

```python
from pathlib import Path
from clinbayes.bayes_engine.loader import load_network

network = load_network(Path("data/networks/mini_test.yaml"))
print(list(network.variables.keys()))
```

## Running Tests

```bash
pytest
```
