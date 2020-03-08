import pytest
from dfa import DFA
from dfa.utils import universal, empty

from probabilistic_automata import lift
from probabilistic_automata.utils import prob_pred, tee
from probabilistic_automata.utils import dict2pdfa, pdfa2dict


def test_dict2pdfa():
    mapping = {
        "s1": (True, {
            'a': {'s1': 0.5, 's2': 0.5},
        }),
        "s2": (False, {
            'a': {'s1': 1},
        }),
    }

    start = "s1"
    pdfa = dict2pdfa(mapping=mapping, start=start)
    assert pdfa.inputs == {'a'}

    mapping2, start2 = pdfa2dict(pdfa)
    assert start == start2
    assert mapping2 == mapping


def test_prob_pred():
    assert prob_pred(lift(universal({0, 1})), pred=bool, horizon=3) == 1
    assert prob_pred(lift(empty({0, 1})), pred=bool, horizon=3) == 0

    or_pdfa = lift(DFA(
        start=0, label=bool, inputs={0, 1}, transition=max
    ))
    assert prob_pred(or_pdfa, pred=bool, horizon=3) == pytest.approx(2**-3)


def test_tee():
    machine = tee(
        lift(universal({0, 1})),
        lift(empty({1})),
    )
    assert machine.inputs == {1}
    assert machine.outputs.left == machine.outputs.right == {False, True}
    assert machine.label((1, 1, 1)) == (True, False)
