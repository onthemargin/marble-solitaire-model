import pytest
import numpy as np
from marble_solitaire.board import (
    BoardState, initial_board, VALID_MASK,
    move_to_index, index_to_move, create_legal_move_mask,
    UP, DOWN, LEFT, RIGHT,
)


class TestBoardInitialization:
    def test_initial_board_has_36_marbles(self):
        board = initial_board()
        assert board.count_marbles() == 36

    def test_center_is_empty(self):
        board = initial_board()
        assert board.grid[3, 3] == 0

    def test_valid_positions_count_37(self):
        assert VALID_MASK.sum() == 37

    def test_invalid_corners(self):
        assert VALID_MASK[0, 0] == 0
        assert VALID_MASK[0, 1] == 0
        assert VALID_MASK[1, 0] == 0
        assert VALID_MASK[6, 5] == 0
        assert VALID_MASK[6, 6] == 0
        assert VALID_MASK[5, 6] == 0

    def test_all_valid_positions_have_marbles_except_center(self):
        board = initial_board()
        for r in range(7):
            for c in range(7):
                if VALID_MASK[r, c] == 1 and (r, c) != (3, 3):
                    assert board.grid[r, c] == 1, f"Expected marble at ({r},{c})"


class TestMoveGeneration:
    def test_initial_board_has_4_legal_moves(self):
        board = initial_board()
        moves = board.get_legal_moves()
        assert len(moves) == 4

    def test_initial_moves_all_jump_to_center(self):
        board = initial_board()
        moves = board.get_legal_moves()
        deltas = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        for r, c, d in moves:
            dr, dc = deltas[d]
            assert (r + 2*dr, c + 2*dc) == (3, 3)


class TestApplyMove:
    def test_apply_move_reduces_marble_count(self):
        board = initial_board()
        moves = board.get_legal_moves()
        new_board = board.apply_move(moves[0])
        assert new_board.count_marbles() == 35

    def test_apply_move_updates_cells(self):
        board = initial_board()
        new_board = board.apply_move((1, 3, DOWN))
        assert new_board.grid[1, 3] == 0
        assert new_board.grid[2, 3] == 0
        assert new_board.grid[3, 3] == 1

    def test_apply_move_is_immutable(self):
        board = initial_board()
        moves = board.get_legal_moves()
        new_board = board.apply_move(moves[0])
        assert board.count_marbles() == 36
        assert new_board.count_marbles() == 35

    def test_apply_move_returns_board_state(self):
        board = initial_board()
        moves = board.get_legal_moves()
        new_board = board.apply_move(moves[0])
        assert isinstance(new_board, BoardState)


class TestTerminal:
    def test_initial_board_is_not_terminal(self):
        board = initial_board()
        assert not board.is_terminal()

    def test_single_marble_is_terminal(self):
        grid = np.zeros((7, 7), dtype=np.int8)
        grid[3, 3] = 1
        board = BoardState(grid)
        assert board.is_terminal()
        assert board.is_solved()

    def test_no_legal_moves_is_terminal(self):
        grid = np.zeros((7, 7), dtype=np.int8)
        grid[0, 2] = 1
        grid[6, 4] = 1
        board = BoardState(grid)
        assert board.is_terminal()
        assert not board.is_solved()


class TestTensor:
    def test_to_tensor_shape(self):
        board = initial_board()
        tensor = board.to_tensor()
        assert tensor.shape == (2, 7, 7)

    def test_tensor_channel0_is_marbles(self):
        board = initial_board()
        tensor = board.to_tensor()
        assert tensor[0, 3, 3] == 0.0
        assert tensor[0, 0, 2] == 1.0
        assert tensor[0, 0, 0] == 0.0

    def test_tensor_channel1_is_valid_mask(self):
        board = initial_board()
        tensor = board.to_tensor()
        np.testing.assert_array_equal(tensor[1], VALID_MASK.astype(np.float32))


class TestMoveEncoding:
    def test_move_to_index_range(self):
        idx = move_to_index(0, 0, UP)
        assert 0 <= idx < 196

    def test_move_index_roundtrip(self):
        for r in range(7):
            for c in range(7):
                for d in range(4):
                    idx = move_to_index(r, c, d)
                    r2, c2, d2 = index_to_move(idx)
                    assert (r, c, d) == (r2, c2, d2)

    def test_create_legal_move_mask_shape(self):
        board = initial_board()
        moves = board.get_legal_moves()
        mask = create_legal_move_mask(moves)
        assert mask.shape == (196,)

    def test_create_legal_move_mask_sum(self):
        board = initial_board()
        moves = board.get_legal_moves()
        mask = create_legal_move_mask(moves)
        assert mask.sum() == len(moves)
