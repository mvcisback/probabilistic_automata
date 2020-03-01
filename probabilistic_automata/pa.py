import random
from collections import defaultdict
from typing import Callable, Hashable, Mapping, Set

import attr
import funcy as fn
from dfa import DFA, SupAlphabet, ProductAlphabet


State = Hashable
Action = Hashable


@attr.s(frozen=True, auto_attribs=True)
class Distribution:
    _dist: Mapping[Action, float]

    def sample(self) -> Action:
        actions, weights = zip(*self._dist.items())
        return random.choices(actions, weights)[0]

    def __call__(self, action):
        return self._dist.get(action, 0)

    def items(self):
        return self._dist.items()


EnvDist = Callable[[State, Action], Distribution]


def uniform(actions: Set[Action]) -> EnvDist:
    size = len(actions)
    dist = Distribution({a: 1/size for a in actions})
    return lambda *_: dist


def _dict2dist(env_dist) -> EnvDist:
    @fn.memoize
    def env_dist2(state, action):
        dist = env_dist(state, action)
        if isinstance(dist, Distribution):
            return dist

        return Distribution(dist)

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
    def check_product_lang(self, _, dfa):
        assert all(isinstance(i, tuple) and len(i) == 2 for i in dfa.inputs)

    @property
    def env_inputs(self):
        return set(fn.pluck(1, self.dfa.inputs))

    @property
    def inputs(self):
        return set(fn.pluck(0, self.dfa.inputs))

    @property
    def outputs(self):
        return self.dfa.outputs

    @property
    def start(self):
        return self.dfa.start

    def states(self):
        return self.dfa.states()

    def run(self, *, start=None, seed=None):
        if seed is not None:
            random.seed(seed)

        state = self.start if start is None else start
        machine = self.dfa.run(start=start)

        while True:
            sys_action = yield state
            env_action = self.env_dist(state, sys_action).sample()
            state = machine.send((sys_action, env_action))

    @fn.memoize
    def support(self, state, action):
        return set(self.transition_probs(state, action).keys())

    def _probs(self, start, action):
        for e, p in self.env_dist(start, action).items():
            end = self.dfa._transition(start, (action, e))
            yield (end, p)

    def transition_probs(self, state, action):
        probs = defaultdict(lambda: 0)
        for end, prob in self._probs(state, action):
            probs[end] += prob
        return probs

    def prob(self, start, end, action):
        return sum(p for s, p in self._probs(start, action) if s == end)


def pdfa(start, label, transition, env_dist,
         inputs=None, env_inputs=None, outputs=None) -> PDFA:

    if inputs is None:
        inputs = SupAlphabet()
    if outputs is None:
        outputs = {True, False}
    if env_inputs is None:
        env_inputs = {None}

    inputs = ProductAlphabet(inputs, env_inputs)

    return PDFA(
        env_dist=env_dist,
        dfa=DFA(
            start=start, label=label,
            inputs=inputs, outputs=outputs,
            transition=transition,
        ),
    )


def lift(dyn: DFA) -> PDFA:
    """Lifts a DFA into a deterministic PDFA."""
    return PDFA(
        dfa=attr.evolve(
            dyn,
            inputs=ProductAlphabet(dyn.inputs, {None}),
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
            inputs=ProductAlphabet({None}, dyn.inputs),
            transition=lambda s, c: dyn._transition(s, c[1]),
        ),
        env_dist=uniform(dyn.inputs),
    )
