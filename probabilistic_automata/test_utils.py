from probabilistic_automata.utils import dict2pdfa


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
