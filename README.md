# Probabilistic Automata

[![Build Status](https://cloud.drone.io/api/badges/mvcisback/probabilistic_automata/status.svg)](https://cloud.drone.io/mvcisback/probabilistic_automata)
[![Docs](https://img.shields.io/badge/API-link-color)](https://mvcisback.github.io/probabilistic_automata)
[![codecov](https://codecov.io/gh/mvcisback/probabilistic_automata/branch/master/graph/badge.svg)](https://codecov.io/gh/mvcisback/probabilistic_automata)
[![PyPI version](https://badge.fury.io/py/probabilistic_automata.svg)](https://badge.fury.io/py/probabilistic_automata)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Python library for manipulating Probabilistic Automata. This library
builds upon the [`dfa`](https://github.com/mvcisback/dfa) package.

# Installation

If you just need to use `probabilistic_automata`, you can just run:

`$ pip install probabilistic_automata`

For developers, note that this project uses the
[poetry](https://poetry.eustace.io/) python package/dependency
management tool. Please familarize yourself with it and then
run:

`$ poetry install`

# Usage

The `probabilistic_automata` library centers around the `PDFA` object
which models a finite probabilistic transition system, e.g., a Markov
Decision Process, as a `DFA` or Moore Machine over a product alphabet
over the system's actions and the environment's stochastic action.

```python
import probabilistic_automata as PA

def transition(state, composite_action):
    sys_action, env_action = composite_action
    return (state + sys_action + env_action) % 2

def env_dist(state, sys_action):
    """Based on state and the system action, what are the probabilities 
    of the environment's action."""

    return {0: 1/2, 1: 1/2}  # Always a coin flip.

noisy_parity = PA.pdfa(
    start=0,
    label=bool,
    inputs={0, 1},
    env_inputs={0, 1},
    outputs={0, 1},
    transition=transition,
    env_dist=env_dist,   # Equivalently, PA.uniform({0, 1}).
)
```

The support and transition probabilities can easily calculated:

```python
assert noisy_parity.support(0, 0) == {0, 1}
assert noisy_parity.transition_probs(0, 0) == {0: 1/2, 1: 1/2}
assert noisy_parity.prob(start=0, action=0, end=0) == 1/2
```

## Dict <-> PDFA

Note that `pdfa` provides helper functions for going from a dictionary
based representation of a probabilistic transition system to a `PDFA`
object and back.

```python
import probabilistic_automata as PA

mapping = {
    "s1": (True, {
        'a': {'s1': 0.5, 's2': 0.5},
    }),
    "s2": (False, {
        'a': {'s1': 1},
    }),
}

start = "s1"
pdfa = PA.dict2pdfa(mapping=mapping, start=start)
assert pdfa.inputs == {'a'}

mapping2, start2 = PA.pdfa2dict(pdfa)
assert start == start2
assert mapping2 == mapping
```


## DFA to PDFA

The `probabilistic_automata` library has two convenience methods for
transforming a Deterministic Finite Automaton (`dfa.DFA`) into a
`PDFA`.

- The `lift` function simply creates a `PDFA` whose transitions are
  deterministic and match the original `dfa.DFA`.

```python
import probabilistic_automata as PA
from dfa import DFA

parity = DFA(
    start=0,
    inputs={0, 1},
    label=bool,
    transition=lambda s, c: (s + c) & 1,
)

parity_pdfa = lift(parity)

assert pdfa.inputs == PARITY.inputs
assert pdfa.env_inputs == {None}
```

- The `randomize` function takes a `DFA` and returns a `PDFA` modeling
  the actions of the `DFA` being selected uniformly at random.

```
pdfa = PA.randomize(PARITY)

assert pdfa.inputs == {None}
assert pdfa.env_inputs == PARITY.inputs
```

