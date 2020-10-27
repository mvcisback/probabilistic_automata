import pytest
from dfa import DFA

from probabilistic_automata import lift
from probabilistic_automata.utils import prob_pred, tee
from probabilistic_automata.utils import dict2pdfa, pdfa2dict


def universal(alphabet):
    return DFA(
        start=True,
        label=lambda *_: True,
        transition=lambda *_: True,
        inputs=alphabet,
    )


def empty(alphabet):
    return DFA(
        start=False,
        label=lambda *_: False,
        transition=lambda *_: False,
        inputs=alphabet,
    )


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
    assert len(machine.states()) == 1
    assert machine.inputs == {1}
    assert machine.outputs == {(0, 0), (0, 1), (1, 0), (1, 1)}
    assert machine.label((1, 1, 1)) == (True, False)

    identity = lift(DFA(
        start=1, label=lambda x: x,
        transition=lambda _, c: c,
        outputs={1}, inputs={1},
    ))
    machine = identity >> machine

    assert len(machine.states()) == 1
    assert machine.inputs == {1}
    assert machine.label((1, 1, 1)) == (True, False)
