from operator import itemgetter as ig

import funcy as fn
from lenses import bind

from probabilistic_automata import pa as PA


def _encode_two_player_game(mapping):
    """Convert transition probabilities from state to two player game
    transition and distribution.

       mapping: State => (Action => (State => Prob))

    where (=>) denotes a Dictionary map.
    """
    e_size = 0  # Environment Alphabet Size.

    def reindex(state2prob):
        """Enumerates states and transforms the mapping:
           State => Prob
        to a mapping:
           Index => (State, Prob)
        """
        nonlocal e_size
        e_size = max(e_size, len(state2prob))

        return {i: (s, p) for i, (s, p) in enumerate(state2prob.items())}

    # State => (Action => (EnvironmentAction => (State, Prob)))
    mapping = bind(mapping).Values().Values().modify(reindex)

    def transition(state, composite_action):
        sys, env = composite_action
        dist = mapping[state][sys]

        if env not in dist:  # Probability 0 event.
            return state
        return dist[env][0]

    def env_dist(state, sys):
        return {e: p for e, (_, p) in mapping[state][sys].items()}

    return {
        "env_inputs": range(e_size),
        "transition": transition,
        "env_dist": env_dist,
    }


def dict2pdfa(mapping, start: PA.State):
    """Convert nested dictionary into a PDFA.

    - mapping is a nested dictionary of the form:

       mapping = {
         <State>:  (<Label>, {
            <Action>: {
                <State>: <Probability>
            }
         }
       }
    """
    label_map = fn.walk_values(ig(0), mapping)
    transition_map = fn.walk_values(ig(1), mapping)

    outputs = set(bind(mapping).Values()[0].collect())
    inputs = set(bind(mapping).Values()[1].Keys().collect())

    return PA.pdfa(
        start=start,
        label=label_map.get,
        inputs=inputs,
        outputs=outputs,
        **_encode_two_player_game(transition_map)
    )


def pdfa2dict(pdfa):
    """Convert PDFA into nested dictionary of the form:

       mapping = {
         <State>:  (<Label>, {
            <Action>: {
                <State>: <Probability>
            }
         }
       }

    Returns: mapping and the start state.
    """
    mapping = {}
    for s in pdfa.states():
        action2state_prob = {
            a: pdfa.transition_probs(s, a) for a in pdfa.inputs
        }

        label = pdfa.dfa._label(s)
        mapping[s] = (label, action2state_prob)
    return mapping, pdfa.start


def prob_pred(dyn, *, pred, horizon) -> float:
    """
    Return the probability that pred will evaluate to true before
    horizon time step assuming system inputs are applied uniformly at
    random.
    """
    prob = {}

    def pevent(state, path_prob, time):
        nonlocal prob
        assert time >= 0

        if state in prob:
            return path_prob * prob[state]
        elif pred(state):
            prob[state] = 0
            return 1
        elif time == 0:
            return 0

        acc = 0
        for action in dyn.inputs:
            for state2, state2_prob in dyn._probs(state, action):
                acc += pevent(state2, path_prob * state2_prob, time - 1)

        return acc / len(dyn.inputs)

    return pevent(dyn.start, path_prob=1, time=horizon)
