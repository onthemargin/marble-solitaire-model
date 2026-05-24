import pytest
import numpy as np
from marble_solitaire.model import SolitaireNet
from marble_solitaire.self_play import run_episode, ReplayBuffer, compute_outcome


class TestComputeOutcome:
    def test_solved_center(self):
        assert compute_outcome(1, center_marble=True) == 1.0

    def test_solved_not_center(self):
        assert compute_outcome(1, center_marble=False) == 0.6

    def test_two_marbles(self):
        assert compute_outcome(2) == pytest.approx(-0.3)

    def test_monotonic(self):
        for n in range(2, 35):
            assert compute_outcome(n) > compute_outcome(n + 1)


class TestRunEpisode:
    @pytest.fixture
    def network(self):
        net = SolitaireNet()
        net.eval()
        return net

    def test_returns_list_of_tuples(self, network):
        examples = run_episode(network, n_simulations=10)
        assert isinstance(examples, list)
        assert len(examples) > 0

    def test_example_shapes(self, network):
        examples = run_episode(network, n_simulations=10)
        state, policy, outcome = examples[0]
        assert state.shape == (2, 7, 7)
        assert policy.shape == (196,)
        assert isinstance(outcome, float)

    def test_policy_sums_to_one(self, network):
        examples = run_episode(network, n_simulations=10)
        for _, policy, _ in examples:
            assert policy.sum() == pytest.approx(1.0, abs=1e-4)

    def test_outcome_in_range(self, network):
        examples = run_episode(network, n_simulations=10)
        for _, _, outcome in examples:
            assert -1.0 <= outcome <= 1.0

    def test_episode_length_plausible(self, network):
        examples = run_episode(network, n_simulations=10)
        assert 1 <= len(examples) <= 35


class TestReplayBuffer:
    def test_add_and_len(self):
        buf = ReplayBuffer(max_size=100)
        examples = [(np.zeros((2, 7, 7)), np.zeros(196), 0.5)] * 10
        buf.add(examples)
        assert len(buf) == 10

    def test_max_size(self):
        buf = ReplayBuffer(max_size=5)
        examples = [(np.zeros((2, 7, 7)), np.zeros(196), 0.5)] * 10
        buf.add(examples)
        assert len(buf) == 5

    def test_sample(self):
        buf = ReplayBuffer(max_size=100)
        examples = [(np.zeros((2, 7, 7)), np.ones(196) / 196, 0.5)] * 20
        buf.add(examples)
        states, policies, outcomes = buf.sample(8)
        assert states.shape == (8, 2, 7, 7)
        assert policies.shape == (8, 196)
        assert outcomes.shape == (8,)
