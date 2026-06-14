"""Board engine for European 37-hole marble solitaire."""

import numpy as np

# Direction constants
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

DIRECTION_DELTAS = {
    UP: (-1, 0),
    DOWN: (1, 0),
    LEFT: (0, -1),
    RIGHT: (0, 1),
}

# European 37-hole board mask (7x7 grid)
# Row 0: cols 2,3,4
# Row 1: cols 1-5
# Rows 2-4: all cols
# Row 5: cols 1-5
# Row 6: cols 2,3,4
VALID_MASK = np.zeros((7, 7), dtype=np.int8)
VALID_MASK[0, 2:5] = 1
VALID_MASK[1, 1:6] = 1
VALID_MASK[2, :] = 1
VALID_MASK[3, :] = 1
VALID_MASK[4, :] = 1
VALID_MASK[5, 1:6] = 1
VALID_MASK[6, 2:5] = 1


class BoardState:
    """Immutable board state for marble solitaire."""

    def __init__(self, grid: np.ndarray):
        self.grid = grid.copy().astype(np.int8)

    def count_marbles(self) -> int:
        return int(self.grid.sum())

    def get_legal_moves(self) -> list[tuple[int, int, int]]:
        """Return list of (row, col, direction) for all legal moves."""
        moves = []
        for r in range(7):
            for c in range(7):
                if self.grid[r, c] != 1:
                    continue
                for d, (dr, dc) in DIRECTION_DELTAS.items():
                    mid_r, mid_c = r + dr, c + dc
                    dst_r, dst_c = r + 2 * dr, c + 2 * dc
                    if (
                        0 <= dst_r < 7
                        and 0 <= dst_c < 7
                        and VALID_MASK[dst_r, dst_c] == 1
                        and self.grid[mid_r, mid_c] == 1
                        and self.grid[dst_r, dst_c] == 0
                    ):
                        moves.append((r, c, d))
        return moves

    def apply_move(self, move: tuple[int, int, int]) -> "BoardState":
        """Apply a move and return a new BoardState (immutable)."""
        r, c, d = move
        dr, dc = DIRECTION_DELTAS[d]
        new_grid = self.grid.copy()
        new_grid[r, c] = 0
        new_grid[r + dr, c + dc] = 0
        new_grid[r + 2 * dr, c + 2 * dc] = 1
        return BoardState(new_grid)

    def is_terminal(self) -> bool:
        return len(self.get_legal_moves()) == 0

    def is_solved(self) -> bool:
        return self.count_marbles() == 1

    def to_tensor(self) -> np.ndarray:
        """Return (2, 7, 7) float32 tensor: [marbles, valid_mask]."""
        tensor = np.zeros((2, 7, 7), dtype=np.float32)
        tensor[0] = self.grid.astype(np.float32)
        tensor[1] = VALID_MASK.astype(np.float32)
        return tensor


def initial_board() -> BoardState:
    """Create the standard European 37-hole starting position."""
    grid = VALID_MASK.copy()
    grid[3, 3] = 0  # center hole empty
    return BoardState(grid)


def move_to_index(r: int, c: int, d: int) -> int:
    """Encode (row, col, direction) as a flat index in [0, 196)."""
    return (r * 7 + c) * 4 + d


def index_to_move(idx: int) -> tuple[int, int, int]:
    """Decode a flat index back to (row, col, direction)."""
    d = idx % 4
    pos = idx // 4
    r = pos // 7
    c = pos % 7
    return (r, c, d)


def create_legal_move_mask(moves: list[tuple[int, int, int]]) -> np.ndarray:
    """Create a binary mask of shape (196,) with 1s at legal move indices."""
    mask = np.zeros(196, dtype=np.float32)
    for r, c, d in moves:
        mask[move_to_index(r, c, d)] = 1.0
    return mask
