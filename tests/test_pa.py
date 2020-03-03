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
    actions = {1, 2, 3}
    dist = PA.uniform(actions)(0, 1)
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


NOISY_PARITY = PA.pdfa(
    start=0,
    label=bool,
    inputs={0, 1},
    env_inputs={0, 1},
    transition=lambda s, c: (s + sum(c)) & 1,
    env_dist=PA.uniform({0, 1}),
)


def test_noisy_parity():
    assert NOISY_PARITY.prob(0, 1, 0) == 1/2
    assert NOISY_PARITY.prob(0, 1, 1) == 1/2
    assert NOISY_PARITY.prob(1, 1, 0) == 1/2
    assert NOISY_PARITY.prob(1, 1, 1) == 1/2


def test_randomize():
    pdfa = PA.randomize(PARITY)

    assert pdfa.inputs == {None}
    assert pdfa.env_inputs == PARITY.inputs
    assert pdfa.outputs == {0, 1}

    for s, a in product(pdfa.states(), pdfa.inputs):
        assert a is None

        support = pdfa.support(s, a)
        assert len(support) == 2
        assert all(pdfa.prob(s, e, a) == 1/2 for e in support)
        assert sum(pdfa.transition_probs(s, a).values()) == 1


def test_run():
    pass


def test_par_compose():
    machine = NOISY_PARITY | NOISY_PARITY

    assert machine.inputs.left == NOISY_PARITY.inputs
    assert machine.inputs.right == NOISY_PARITY.inputs

    assert machine.outputs.left == NOISY_PARITY.outputs
    assert machine.outputs.right == NOISY_PARITY.outputs

    assert machine.env_inputs.left == NOISY_PARITY.env_inputs
    assert machine.env_inputs.right == NOISY_PARITY.env_inputs

    assert machine.start == (0, 0)
    assert machine.support((0, 0), (0, 0)) == {(0, 0), (0, 1), (1, 0), (1, 1)}

    # Deterministic | Noisy
    machine = PA.lift(PARITY) | NOISY_PARITY
    assert machine.support((0, 0), (0, 0)) == {(0, 0), (0, 1)}
    assert machine.support((0, 0), (1, 0)) == {(1, 0), (1, 1)}
    assert machine.support((1, 0), (0, 0)) == {(1, 0), (1, 1)}


def test_seq_compose():
    # Noisy >> Noisy
    machine = NOISY_PARITY >> NOISY_PARITY

    assert machine.inputs == NOISY_PARITY.inputs
    assert machine.outputs == NOISY_PARITY.outputs
    assert machine.start == (0, 0)

    assert machine.support((0, 0), 0) == {(0, 0), (0, 1), (1, 0), (1, 1)}

    # Deterministic >> Noisy
    machine = PA.lift(PARITY) >> NOISY_PARITY

    assert machine.inputs == NOISY_PARITY.inputs
    assert machine.outputs == NOISY_PARITY.outputs
    assert machine.start == (0, 0)

    assert machine.support((0, 0), 0) == {(0, 0), (0, 1)}
    assert machine.support((0, 0), 1) == {(1, 0), (1, 1)}
    assert machine.support((1, 0), 0) == {(1, 0), (1, 1)}

    # Deterministic >> Deterministic
    machine = PA.lift(PARITY) >> PA.lift(PARITY)

    assert machine.start == (0, 0)
    assert machine.support((0, 0), 0) == {(0, 0)}
    assert machine.support((0, 0), 1) == {(1, 0)}
    assert machine.support((1, 0), 0) == {(1, 1)}
    assert machine.support((1, 1), 0) == {(1, 0)}
    assert machine.support((1, 1), 1) == {(0, 0)}
