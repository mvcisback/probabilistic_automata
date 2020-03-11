from typing import Sequence

import attr

from probabilistic_automata.pa import PDFA, State


Trace = Sequence[State]


@attr.s(frozen=True, auto_attribs=True)
class MarkovChain(PDFA):
    def __attrs_post_init__(self):
        """
        Check that PDFA only has a single system input, implying
        that the resulting PDFA's transitions are only governed by
        environment inputs.
        """
        if len(self.inputs) != 1:
            raise ValueError(
                "Number of system inputs != 1\n"
                "Underlying system potentially non-determinstic."
            )

    def trace_prob(self, trc: Trace) -> float:
        """
        Compute the probability of this markov chain generating the
        provided trace.
        """
        sys_input, *_ = self.inputs
        prev_state, prob = self.start, 1
        for state in trc:
            prob *= self.prob(prev_state, state, sys_input)
            if prob == 0:
                return 0
        return prob


def from_pdfa(pdfa: PDFA) -> MarkovChain:
    return MarkovChain(dfa=pdfa.dfa, env_dist=pdfa.env_dist)
