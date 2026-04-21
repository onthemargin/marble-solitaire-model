// European 37-hole board layout
const VALID_MASK: number[][] = [
  [0, 0, 1, 1, 1, 0, 0],
  [0, 1, 1, 1, 1, 1, 0],
  [1, 1, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1, 1, 1],
  [0, 1, 1, 1, 1, 1, 0],
  [0, 0, 1, 1, 1, 0, 0],
];

const CELL_SIZE = 44;
const MARGIN = 22;
const MARBLE_R = 18;

export type BoardGrid = number[][];

export interface Move {
  row: number;
  col: number;
  dir: number; // 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
}

const DIR_DELTAS: Record<number, [number, number]> = {
  0: [-1, 0], // UP
  1: [1, 0],  // DOWN
  2: [0, -1], // LEFT
  3: [0, 1],  // RIGHT
};

export function initialBoard(): BoardGrid {
  const grid: BoardGrid = [];
  for (let r = 0; r < 7; r++) {
    grid[r] = [];
    for (let c = 0; c < 7; c++) {
      if (r === 3 && c === 3) grid[r][c] = 0;
      else grid[r][c] = VALID_MASK[r][c];
    }
  }
  return grid;
}

export function countMarbles(grid: BoardGrid): number {
  let count = 0;
  for (let r = 0; r < 7; r++)
    for (let c = 0; c < 7; c++)
      if (VALID_MASK[r][c] && grid[r][c]) count++;
  return count;
}

export function getLegalMoves(grid: BoardGrid): Move[] {
  const moves: Move[] = [];
  for (let r = 0; r < 7; r++) {
    for (let c = 0; c < 7; c++) {
      if (!VALID_MASK[r][c] || !grid[r][c]) continue;
      for (const [dir, [dr, dc]] of Object.entries(DIR_DELTAS)) {
        const mr = r + dr, mc = c + dc;
        const lr = r + 2 * dr, lc = c + 2 * dc;
        if (lr >= 0 && lr < 7 && lc >= 0 && lc < 7
          && VALID_MASK[mr][mc] && grid[mr][mc]
          && VALID_MASK[lr][lc] && !grid[lr][lc]) {
          moves.push({ row: r, col: c, dir: Number(dir) });
        }
      }
    }
  }
  return moves;
}

export function applyMove(grid: BoardGrid, move: Move): BoardGrid {
  const newGrid = grid.map(row => [...row]);
  const [dr, dc] = DIR_DELTAS[move.dir];
  newGrid[move.row][move.col] = 0;
  newGrid[move.row + dr][move.col + dc] = 0;
  newGrid[move.row + 2 * dr][move.col + 2 * dc] = 1;
  return newGrid;
}

function cx(c: number): number { return MARGIN + c * CELL_SIZE + CELL_SIZE / 2; }
function cy(r: number): number { return MARGIN + r * CELL_SIZE + CELL_SIZE / 2; }

export function renderBoard(svg: SVGSVGElement, grid: BoardGrid): void {
  svg.innerHTML = '';
  for (let r = 0; r < 7; r++) {
    for (let c = 0; c < 7; c++) {
      if (!VALID_MASK[r][c]) continue;
      // Hole
      const hole = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
      hole.setAttribute('cx', String(cx(c)));
      hole.setAttribute('cy', String(cy(r)));
      hole.setAttribute('r', String(MARBLE_R));
      hole.setAttribute('class', 'hole');
      svg.appendChild(hole);
      // Marble
      if (grid[r][c]) {
        const marble = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        marble.setAttribute('cx', String(cx(c)));
        marble.setAttribute('cy', String(cy(r)));
        marble.setAttribute('r', String(MARBLE_R - 2));
        marble.setAttribute('class', 'marble');
        marble.setAttribute('data-r', String(r));
        marble.setAttribute('data-c', String(c));
        svg.appendChild(marble);
      }
    }
  }
}

export function animateMove(
  svg: SVGSVGElement,
  grid: BoardGrid,
  move: Move,
  onComplete: () => void,
  duration: number = 400,
): void {
  const [dr, dc] = DIR_DELTAS[move.dir];
  const srcR = move.row, srcC = move.col;
  const midR = srcR + dr, midC = srcC + dc;
  const dstR = srcR + 2 * dr, dstC = srcC + 2 * dc;

  // Find the marble being moved
  const marbles = svg.querySelectorAll('.marble');
  let movingMarble: SVGCircleElement | null = null;
  let jumpedMarble: SVGCircleElement | null = null;

  marbles.forEach((m) => {
    const el = m as SVGCircleElement;
    const r = Number(el.getAttribute('data-r'));
    const c = Number(el.getAttribute('data-c'));
    if (r === srcR && c === srcC) movingMarble = el;
    if (r === midR && c === midC) jumpedMarble = el;
  });

  if (!movingMarble || !jumpedMarble) {
    onComplete();
    return;
  }

  // Animate the jump
  movingMarble.style.transition = `cx ${duration}ms ease-in-out, cy ${duration}ms ease-in-out`;
  movingMarble.setAttribute('cx', String(cx(dstC)));
  movingMarble.setAttribute('cy', String(cy(dstR)));
  movingMarble.setAttribute('data-r', String(dstR));
  movingMarble.setAttribute('data-c', String(dstC));

  // Fade out jumped marble
  setTimeout(() => {
    if (jumpedMarble) {
      jumpedMarble.style.transition = `opacity ${duration / 2}ms`;
      jumpedMarble.style.opacity = '0';
    }
  }, duration / 3);

  setTimeout(() => {
    if (jumpedMarble) jumpedMarble.remove();
    if (movingMarble) movingMarble.style.transition = '';
    onComplete();
  }, duration);
}

export function boardToTensor(grid: BoardGrid): Float32Array {
  // (2, 7, 7) flattened = 98 floats
  const tensor = new Float32Array(2 * 7 * 7);
  for (let r = 0; r < 7; r++) {
    for (let c = 0; c < 7; c++) {
      // Channel 0: marbles
      tensor[r * 7 + c] = (VALID_MASK[r][c] && grid[r][c]) ? 1 : 0;
      // Channel 1: valid mask
      tensor[49 + r * 7 + c] = VALID_MASK[r][c];
    }
  }
  return tensor;
}
