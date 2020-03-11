import attr

from probabilistic_automata.pa import PDFA


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


def from_pdfa(pdfa: PDFA) -> MarkovChain:
    return MarkovChain(dfa=pdfa.dfa, env_dist=pdfa.env_dist)
