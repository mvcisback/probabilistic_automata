from __future__ import annotations

import random
from typing import Callable, Mapping, Set, Union
from itertools import product

import attr

from dfa.dfa import Letter, State


Action = Letter


@attr.s(frozen=True, auto_attribs=True)
class ExplicitDistribution:
    """Object representing a discrete distribution over environment actions."""
    _dist: Mapping[Action, float]

    def sample(self) -> Action:
        """Sample an envionment action."""
        actions, weights = zip(*self._dist.items())
        return random.choices(actions, weights)[0]

    def __call__(self, action):
        """Evaluates the probability of an action."""
        return self._dist.get(action, 0)

    def items(self):
        """Sequence of Action, Probability pairs defining the distribution."""
        return self._dist.items()


@attr.s(frozen=True, auto_attribs=True)
class ProductDistribution:
    left: Distribution
    right: Distribution

    def sample(self) -> Action:
        """Sample an envionment action."""
        return (self.left.sample(), self.right.sample())

    def __call__(self, action):
        """Evaluates the probability of an action."""
        left_a, right_a = action
        return self.left(left_a), self.right(right_a)

    def items(self):
        """Sequence of Action, Probability pairs defining the distribution."""
        prod = product(self.left.items(), self.right.items())
        for (a1, p1), (a2, p2) in prod:
            yield (a1, a2), p1 * p2


Distribution = Union[ProductDistribution, ExplicitDistribution]
EnvDist = Callable[[State, Action], Distribution]


def prod_dist(left: EnvDist, right: EnvDist) -> EnvDist:
    return lambda s, a: ProductDistribution(
        left=left(s[0], a[0]),
        right=right(s[1], a[1]),
    )


def uniform(actions: Set[Action]) -> EnvDist:
    """
    Encodes an environment that selects actions uniformly at random,
    i.e., maps all state/action combinations to a Uniform distribution
    of the input (environment) actions.
    """
    size = len(actions)
    dist = ExplicitDistribution({a: 1/size for a in actions})
    return lambda *_: dist
