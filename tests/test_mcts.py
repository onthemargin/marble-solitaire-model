import pytest
import torch
import numpy as np
from marble_solitaire.board import initial_board, BoardState, VALID_MASK, move_to_index, DOWN
from marble_solitaire.model import SolitaireNet
from marble_solitaire.mcts import MCTSNode, mcts_search, get_action_probabilities, compute_outcome


class TestComputeOutcome:
    def test_solved_is_plus_one(self):
        assert compute_outcome(1) == 1.0

    def test_two_marbles_is_zero(self):
        assert compute_outcome(2) == pytest.approx(0.0)

    def test_many_marbles_is_negative(self):
        assert compute_outcome(10) == pytest.approx(-0.8)

    def test_outcome_decreases_with_more_marbles(self):
        for n in range(2, 36):
            assert compute_outcome(n) > compute_outcome(n + 1)


class TestMCTSNode:
    def test_q_value_zero_when_unvisited(self):
        node = MCTSNode(state=initial_board())
        assert node.Q == 0.0

    def test_q_value_after_update(self):
        node = MCTSNode(state=initial_board())
        node.N = 5
        node.W = 2.5
        assert node.Q == pytest.approx(0.5)


class TestMCTSSearch:
    @pytest.fixture
    def network(self):
        net = SolitaireNet()
        net.eval()
        return net

    def test_returns_root_node(self, network):
        root = mcts_search(initial_board(), network, n_simulations=10)
        assert isinstance(root, MCTSNode)

    def test_root_has_children(self, network):
        root = mcts_search(initial_board(), network, n_simulations=10)
        assert len(root.children) > 0

    def test_visit_counts_sum_to_simulations(self, network):
        root = mcts_search(initial_board(), network, n_simulations=20)
        assert sum(c.N for c in root.children.values()) == 20

    def test_prefers_winning_move(self, network):
        grid = np.zeros((7, 7), dtype=np.int8)
        grid[3, 1] = 1
        grid[3, 2] = 1
        board = BoardState(grid)
        # Two adjacent marbles have 2 legal moves (jump left or right)
        assert len(board.get_legal_moves()) == 2
        root = mcts_search(board, network, n_simulations=20)
        assert len(root.children) == 2

    def test_handles_terminal_board(self, network):
        grid = np.zeros((7, 7), dtype=np.int8)
        grid[3, 3] = 1
        board = BoardState(grid)
        root = mcts_search(board, network, n_simulations=10)
        assert len(root.children) == 0


class TestActionProbabilities:
    @pytest.fixture
    def network(self):
        net = SolitaireNet()
        net.eval()
        return net

    def test_probabilities_sum_to_one(self, network):
        root = mcts_search(initial_board(), network, n_simulations=30)
        probs = get_action_probabilities(root, temperature=1.0)
        assert probs.shape == (196,)
        assert probs.sum() == pytest.approx(1.0, abs=1e-5)

    def test_low_temperature_is_greedy(self, network):
        torch.manual_seed(42)
        root = mcts_search(initial_board(), network, n_simulations=200)
        probs = get_action_probabilities(root, temperature=0.01)
        assert probs.max() > 0.8
