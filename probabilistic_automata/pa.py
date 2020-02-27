import random
from typing import Callable, Hashable, Mapping, Optional, Set

import attr
import funcy as fn
from dfa import DFA


State = Hashable
Action = Hashable


@attr.s(frozen=True, auto_attribs=True)
class Distribution:
    _dist: Mapping[Action, float]

    def sample(self) -> Action:
        actions, weights = zip(*self._dist.items())
        return random.choices(actions, weights)[0]

    def __call__(self, action):
        return self._dist[action]


def uniform(actions: Set[Action])-> Distribution:
    size = len(actions)
    return Distribution({a: 1/size for a in actions})


State2ActionDist = Callable[[State], Distribution]


def _dict2dist(state2dist) -> State2ActionDist:
    @fn.memoize
    def s2d(state):
        dist = state2dist(state)
        if isinstance(dist, Distribution):
            return dist

        return Distribution(dist)

    return s2d


@attr.s(frozen=True, auto_attribs=True)
class PDFA:
    """A DFA over a product alphabet where the first value
    is non-deterministic and the second value is set according
    to a state indexed stationary distribution.
    """
    dfa: DFA = attr.ib()
    state2dist: State2ActionDist = attr.ib(converter=_dict2dist)

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

    def states(self):
        return self.dfa.states()

    def run(self, *, start=None, seed=None):
        if seed is not None:
            random.seed(seed)

        state = self.start if start is None else start
        machine = self.dfa.run(start=start)

        while True:
            action = yield state
            action2 = (action, self.state2dist(state).sample())
            state = machine.send(action2)

    @fn.memoize
    def support(self, state, action):
        actions = {(action, e) for e in self.env_inputs}
        return {self.dfa._transition(state, a) for a in actions}

    def prob(self, start, end, action):
        def reach_end(e):
            return self.dfa._transition(start, (action, e)) == end

        dist = self.state2dist(start)
        return sum(dist(e) for e in self.env_inputs if reach_end(e))


def lift(dyn: DFA) -> PDFA:
    return PDFA(
        dfa=attr.evolve(
            dyn,
            inputs={(i, None) for i in dyn.inputs},
            transition=lambda s, c: dyn._transition(s, c[0]),
        ),
        state2dist=lambda _: uniform({None}),
    )
