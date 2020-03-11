import pytest

from dfa import DFA
from probabilistic_automata import pa as PA
from probabilistic_automata import markov_chain as MC


def test_validator():
    sys = DFA(
        start=0, inputs={0, 1},
        label=lambda s: s == 2,
        transition=lambda s, c: s + sum(c) % 4,
    )

    with pytest.raises(ValueError):
        MC.from_pdfa(PA.lift(sys))   # Non-determinism.

    MC.from_pdfa(PA.randomize(sys))  # Actually Markov Chain.
