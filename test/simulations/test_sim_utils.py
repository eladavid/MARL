import pytest
from fixtures import states, actions


def test_get_all_deterministic_policies(states, actions):
    assert 1 == 0