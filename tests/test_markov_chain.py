import pytest

from dfa import DFA
from probabilistic_automata import pa as PA
from probabilistic_automata import markov_chain as MC


SYS = DFA(
    start=0, inputs={0, 1},
    label=lambda s: s == 2,
    transition=lambda s, c: (s + c) % 4,
)


def test_validator():
    with pytest.raises(ValueError):
        MC.from_pdfa(PA.lift(SYS))   # Non-determinism.

    MC.from_pdfa(PA.randomize(SYS))  # Actually Markov Chain.


def test_trace_prob():
    mc1 = MC.from_pdfa(PA.randomize(SYS))

    trc = 3*[0]
    assert pytest.approx(2**-3) == mc1.trace_prob(trc)

    trc = 3*[1, 2, 1]
    assert 0 == mc1.trace_prob(trc)
