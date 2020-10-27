"""Code for modeling a probablistic automaton."""
from __future__ import annotations

import random
from collections import defaultdict
from itertools import product
from typing import Callable, Mapping, Set, Union

import attr
import funcy as fn
from dfa import DFA
from dfa.dfa import Alphabet, Letter, State

from probabilistic_automata.distributions import (
    Action, EnvDist, Distribution, ProductDistribution,
    prod_dist, uniform
)


def prod_alphabet(left, right) -> Alphabet:
    return frozenset(product(left, right))


def _dict2dist(env_dist) -> EnvDist:
    @fn.memoize
    def env_dist2(state, action):
        dist = env_dist(state, action)
        if isinstance(dist, Distribution.__args__):
            return dist
        return dist

    return env_dist2


@attr.s(frozen=True, auto_attribs=True)
class PDFA:
    """A DFA over a product alphabet where the first value
    is non-deterministic and the second value is set according
    to a state indexed stationary distribution.
    """
    dfa: DFA = attr.ib()
    env_dist: EnvDist = attr.ib(converter=_dict2dist)

    @dfa.validator
    def _check_product_lang(self, _, dfa):
        assert all(isinstance(i, tuple) and len(i) == 2 for i in dfa.inputs)

    @property
    def env_inputs(self):
        """Accesses the set of environment inputs."""
        return set(fn.pluck(1, self.dfa.inputs))

    @property
    def inputs(self):
        """Accesses the set of (non-environment) inputs."""
        return set(fn.pluck(0, self.dfa.inputs))

    @property
    def outputs(self):
        """Accesses the set of possible outputs."""
        return self.dfa.outputs

    @property
    def start(self):
        """Accesses the start state."""
        return self.dfa.start

    def states(self):
        """Computes the set of reachable states from start."""
        return self.dfa.states()

    def _label(self, state):
        return self.dfa.label((), start=state)

    def run(self, *, start=None, seed=None, label=False):
        """Co-routine interface for simulating runs of the automaton.

        - Users can send system actions (elements of self.inputs).
        - Co-routine yields the current state.

        If label is True, then state labels are returned instead
        of states.

        Example:
        =======

        machine: PDFA = ..
        my_input: Action = ..            # Element of machine.inputs.

        sim = machine.run()              # Start co-routine.

        state1 = sim.send(my_input)
        state2 = sim.send(my_input)
        """
        if seed is not None:
            random.seed(seed)

        labeler = self.dfa._label if label else lambda x: x

        machine = self.dfa.run(start=start)
        next(machine)
        state = self.dfa.start

        while True:
            sys_action = yield labeler(state)
            env_action = self.env_dist(state, sys_action).sample()
            state = machine.send((sys_action, env_action))

    def trace(self, word, *, start=None):
        machine = self.run(start=start)
        next(machine)
        for letter in word:
            yield machine.send(letter)

    transition = DFA.transition
    transduce = DFA.transduce
    label = DFA.label

    @fn.memoize
    def support(self, state, action) -> Set[State]:
        """Returns the set of reachable states given (state, action)."""
        return set(self.transition_probs(state, action).keys())

    def _probs(self, start, action):
        for e, p in self.env_dist(start, action).items():
            end = self.dfa._transition(start, (action, e))
            yield (end, p)

    def transition_probs(self, state, action) -> Mapping[State, float]:
        """Returns distribution over states given (state, action)"""
        probs = defaultdict(lambda: 0)
        for end, prob in self._probs(state, action):
            probs[end] += prob
        return probs

    def prob(self, start: State, end: State, action: Action) -> float:
        """
        Returns the probability of transitioning from start to end
        given action.
        """
        return sum(p for s, p in self._probs(start, action) if s == end)

    def __or__(self, other: PDFA) -> PDFA:
        def transition(composite_state, composite_action):
            state_l, state_r = composite_state
            (input_l, input_r), (env_l, env_r) = composite_action

            state2_l = self.dfa._transition(state_l, (input_l, env_l))
            state2_r = other.dfa._transition(state_r, (input_r, env_r))
            return (state2_l, state2_r)

        return pdfa(
            start=(self.start, other.start),
            label=lambda s: (self.dfa._label(s[0]), other.dfa._label(s[1])),
            transition=transition,
            env_dist=prod_dist(self.env_dist, other.env_dist),
            inputs=prod_alphabet(self.inputs, other.inputs),
            env_inputs=prod_alphabet(self.env_inputs, other.env_inputs),
            outputs=prod_alphabet(self.outputs, other.outputs),
        )

    def __rshift__(self, other: PDFA) -> PDFA:
        def transition(composite_state, composite_action):
            state_l, state_r = composite_state
            input_l, (env_l, env_r) = composite_action
            input_r = self.dfa._label(state_l)

            state2_l = self.dfa._transition(state_l, (input_l, env_l))
            state2_r = other.dfa._transition(state_r, (input_r, env_r))

            return (state2_l, state2_r)

        def env_dist(composite_state, input_l):
            state_l, state_r = composite_state
            input_r = self.dfa._label(state_l)

            return ProductDistribution(
                left=self.env_dist(state_l, input_l),
                right=other.env_dist(state_r, input_r),
            )

        return pdfa(
            start=(self.start, other.start),
            label=lambda s: other.dfa._label(s[1]),
            transition=transition,
            env_dist=env_dist,
            inputs=self.inputs,
            env_inputs=prod_alphabet(self.env_inputs, other.env_inputs),
            outputs=other.outputs,
        )

    def __lshift__(self, other: PDFA) -> PDFA:
        return other >> self


def pdfa(
        start: State,
        label: Callable[[State], Letter],
        transition: Callable[[State, Action], State],
        env_dist: EnvDist = None,
        inputs: Alphabet = None,
        env_inputs: Alphabet = None,
        outputs: Alphabet = None
) -> PDFA:
    """Main entrypoint for construction a Probablistic Automaton."""

    if outputs is None:
        outputs = {True, False}
    if env_inputs is None and env_dist is None:
        env_inputs = {None}
        env_dist = uniform(env_inputs)

    inputs = None if inputs is None else set(product(inputs, env_inputs))

    return PDFA(
        env_dist=env_dist,
        dfa=DFA(
            start=start, label=label,
            inputs=inputs, outputs=outputs,
            transition=transition,
        ),
    )


def lift(dyn: Union[DFA, PDFA]) -> PDFA:
    """Lifts a DFA into a deterministic PDFA."""
    if isinstance(dyn, PDFA):
        return dyn

    return PDFA(
        dfa=attr.evolve(
            dyn,
            inputs=prod_alphabet(dyn.inputs, {None}),
            transition=lambda s, c: dyn._transition(s, c[0]),
        ),
        env_dist=uniform({None}),
    )


def randomize(dyn: DFA) -> PDFA:
    """Lifts a DFA into a PDFA where original inputs are applied
    uniformly at random.
    """
    return PDFA(
        dfa=attr.evolve(
            dyn,
            inputs=prod_alphabet({None}, dyn.inputs),
            transition=lambda s, c: dyn._transition(s, c[1]),
        ),
        env_dist=uniform(dyn.inputs),
    )
