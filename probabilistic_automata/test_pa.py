from itertools import product

from dfa import DFA

from probabilistic_automata import pa as PA


PARITY = DFA(
    start=0,
    inputs={0, 1},
    label=bool,
    transition=lambda s, c: (s + c) & 1,
)


def test_uniform_dist_smoke():
    actions = {1,2,3}
    dist = PA.uniform(actions)
    assert dist.sample() in actions

    for a in actions:
        assert dist(a) == 1/3


def test_lift():
    pdfa = PA.lift(PARITY)

    assert pdfa.inputs == PARITY.inputs
    assert pdfa.env_inputs == {None}
    assert pdfa.outputs == {0, 1}

    for s, a in product(pdfa.states(), pdfa.inputs):
        end, *other = pdfa.support(s, a)
        assert len(other) == 0
        assert pdfa.prob(s, end, a) == 1
