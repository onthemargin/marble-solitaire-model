import { initialBoard, renderBoard, renderBoardWithSelection, animateMove, applyMove, countMarbles, getLegalMoves, findMove, getClickTarget, BoardGrid, Move } from './board';
import { loadModel, predictMove, isModelLoaded } from './solver';

let currentGrid: BoardGrid = initialBoard();
let moveHistory: Move[] = [];
let solving = false;
let manualMode = false;
let selectedMarble: { row: number; col: number } | null = null;
let currentGen = 1;

const svg = document.getElementById('board') as unknown as SVGSVGElement;
const playArea = document.querySelector('.play-area') as HTMLElement;
const marbleCountEl = document.getElementById('marble-count')!;
const moveCountEl = document.getElementById('move-count')!;
const confidenceEl = document.getElementById('confidence')!;
const statusEl = document.getElementById('status')!;
const solveBtn = document.getElementById('solve-btn') as HTMLButtonElement;
const resetBtn = document.getElementById('reset-btn') as HTMLButtonElement;
const speedSlider = document.getElementById('speed') as HTMLInputElement;
const genBtns = document.querySelectorAll('.gen-btn');
const modeBtns = document.querySelectorAll<HTMLButtonElement>('.mode-btn');

function updateStats(confidence?: number) {
  marbleCountEl.textContent = String(countMarbles(currentGrid));
  moveCountEl.textContent = String(moveHistory.length);
  confidenceEl.textContent = confidence !== undefined ? `${(confidence * 100).toFixed(0)}%` : '--';
}

function setStatus(text: string) {
  statusEl.textContent = text;
}

function getSpeed(): number {
  const min = Number(speedSlider.min);
  const max = Number(speedSlider.max);
  return max + min - Number(speedSlider.value);
}

function getValidTargets(grid: BoardGrid, fromR: number, fromC: number): Set<string> {
  const targets = new Set<string>();
  const moves = getLegalMoves(grid);
  for (const m of moves) {
    if (m.row === fromR && m.col === fromC) {
      const [dr, dc] = [m.dir === 1 ? 1 : m.dir === 0 ? -1 : 0, m.dir === 3 ? 1 : m.dir === 2 ? -1 : 0];
      targets.add(`${fromR + 2 * dr},${fromC + 2 * dc}`);
    }
  }
  return targets;
}

async function selectGen(gen: number) {
  currentGen = gen;
  genBtns.forEach(btn => {
    btn.classList.toggle('active', Number((btn as HTMLElement).dataset.gen) === gen);
  });
  setStatus('Loading...');
  solveBtn.disabled = true;
  try {
    await loadModel(gen);
    setStatus(manualMode ? 'Your turn' : 'Ready');
    solveBtn.disabled = false;
  } catch (e) {
    setStatus(manualMode ? 'Your turn' : 'Model not found');
    console.error(e);
  }
}

function setMode(mode: 'ai' | 'manual') {
  if (solving) return;
  manualMode = mode === 'manual';
  selectedMarble = null;
  playArea.dataset.mode = mode;
  modeBtns.forEach(btn => {
    const active = btn.dataset.mode === mode;
    btn.classList.toggle('active', active);
    btn.setAttribute('aria-selected', String(active));
  });
  // Reset the board when switching modes so each mode starts fresh.
  currentGrid = initialBoard();
  moveHistory = [];
  renderBoard(svg, currentGrid);
  updateStats();
  if (manualMode) {
    setStatus('Your turn — click a marble');
    svg.style.cursor = 'pointer';
  } else {
    setStatus(isModelLoaded() ? 'Ready' : 'Loading...');
    solveBtn.disabled = !isModelLoaded();
    svg.style.cursor = '';
  }
}

function reset() {
  solving = false;
  currentGrid = initialBoard();
  moveHistory = [];
  selectedMarble = null;
  renderBoard(svg, currentGrid);
  updateStats();
  if (manualMode) {
    setStatus('Your turn — click a marble');
  } else {
    setStatus('Ready');
    solveBtn.disabled = !isModelLoaded();
  }
  solveBtn.textContent = 'Solve';
}

async function handleBoardClick(e: MouseEvent) {
  if (!manualMode || solving) return;

  // Use SVG's native coordinate transform — works across browsers/DPR/touch.
  const pt = svg.createSVGPoint();
  pt.x = e.clientX;
  pt.y = e.clientY;
  const ctm = svg.getScreenCTM();
  if (!ctm) return;
  const svgPt = pt.matrixTransform(ctm.inverse());

  const target = getClickTarget(svgPt.x, svgPt.y);
  if (!target) {
    selectedMarble = null;
    renderBoard(svg, currentGrid);
    setStatus('Your turn — click a marble');
    return;
  }

  if (selectedMarble) {
    // Try to make a move to the clicked position
    const move = findMove(currentGrid, selectedMarble.row, selectedMarble.col, target.row, target.col);
    if (move) {
      const newGrid = applyMove(currentGrid, move);
      moveHistory.push(move);

      await new Promise<void>(resolve => {
        animateMove(svg, currentGrid, move, resolve, 300);
      });

      currentGrid = newGrid;
      selectedMarble = null;
      renderBoard(svg, currentGrid);
      updateStats();

      const legal = getLegalMoves(currentGrid);
      if (legal.length === 0) {
        const remaining = countMarbles(currentGrid);
        setStatus(remaining === 1 ? 'You solved it!' : `No moves left (${remaining} remaining)`);
      } else {
        setStatus('Your turn — click a marble');
      }
      return;
    }

    // Clicked a different marble — reselect
    if (currentGrid[target.row][target.col]) {
      selectedMarble = target;
      const targets = getValidTargets(currentGrid, target.row, target.col);
      if (targets.size > 0) {
        renderBoardWithSelection(svg, currentGrid, target.row, target.col, targets);
        setStatus('Click a highlighted hole to jump');
      } else {
        renderBoardWithSelection(svg, currentGrid, target.row, target.col);
        setStatus('No jumps from here — try another');
      }
      return;
    }

    // Clicked an invalid target
    selectedMarble = null;
    renderBoard(svg, currentGrid);
    setStatus('Your turn — click a marble');
  } else {
    // Select a marble
    if (currentGrid[target.row][target.col]) {
      selectedMarble = target;
      const targets = getValidTargets(currentGrid, target.row, target.col);
      if (targets.size > 0) {
        renderBoardWithSelection(svg, currentGrid, target.row, target.col, targets);
        setStatus('Click a highlighted hole to jump');
      } else {
        renderBoardWithSelection(svg, currentGrid, target.row, target.col);
        setStatus('No jumps from here — try another');
      }
    }
  }
}

async function solve() {
  if (solving) {
    solving = false;
    solveBtn.textContent = 'Solve';
    return;
  }

  if (!isModelLoaded()) {
    setStatus('No model loaded');
    return;
  }

  solving = true;
  solveBtn.textContent = 'Stop';
  resetBtn.disabled = true;
  modeBtns.forEach(b => (b.disabled = true));
  setStatus('Solving...');

  while (solving) {
    const legal = getLegalMoves(currentGrid);
    if (legal.length === 0) {
      const remaining = countMarbles(currentGrid);
      setStatus(remaining === 1 ? 'Solved!' : `Stuck (${remaining} left)`);
      break;
    }

    const result = await predictMove(currentGrid);
    if (!result) {
      setStatus('No move found');
      break;
    }

    const { move, confidence } = result;
    const newGrid = applyMove(currentGrid, move);
    moveHistory.push(move);

    await new Promise<void>(resolve => {
      animateMove(svg, currentGrid, move, resolve, getSpeed());
    });

    currentGrid = newGrid;
    renderBoard(svg, currentGrid);
    updateStats(confidence);

    await new Promise(r => setTimeout(r, 50));
  }

  solving = false;
  solveBtn.textContent = 'Solve';
  resetBtn.disabled = false;
  modeBtns.forEach(b => (b.disabled = false));
}

// Wire up events
solveBtn.addEventListener('click', solve);
resetBtn.addEventListener('click', reset);
modeBtns.forEach(btn => {
  btn.addEventListener('click', () => {
    const mode = btn.dataset.mode as 'ai' | 'manual';
    setMode(mode);
  });
});
svg.addEventListener('click', handleBoardClick);
genBtns.forEach(btn => {
  btn.addEventListener('click', () => {
    const gen = Number((btn as HTMLElement).dataset.gen);
    reset();
    selectGen(gen);
  });
});

// Initial render — start in AI Mode
playArea.dataset.mode = 'ai';
renderBoard(svg, currentGrid);
updateStats();
selectGen(1);
