import pytest
import torch
from marble_solitaire.board import initial_board, BoardState, VALID_MASK
from marble_solitaire.model import SolitaireNet
from marble_solitaire.inference import solve_greedy
import numpy as np


class TestSolveGreedy:
    @pytest.fixture
    def network(self):
        net = SolitaireNet()
        net.eval()
        return net

    def test_returns_move_list(self, network):
        board = initial_board()
        moves = solve_greedy(network, board)
        assert isinstance(moves, list)
        assert len(moves) > 0

    def test_all_moves_valid(self, network):
        board = initial_board()
        moves = solve_greedy(network, board)
        current = board
        for move in moves:
            legal = current.get_legal_moves()
            assert move in legal, f"Move {move} not legal"
            current = current.apply_move(move)

    def test_reduces_marble_count(self, network):
        board = initial_board()
        moves = solve_greedy(network, board)
        result = board
        for m in moves:
            result = result.apply_move(m)
        assert result.count_marbles() < 36
