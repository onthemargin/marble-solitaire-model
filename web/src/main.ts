import { initialBoard, renderBoard, animateMove, applyMove, countMarbles, getLegalMoves, BoardGrid, Move } from './board';
import { loadModel, predictMove, isModelLoaded } from './solver';

let currentGrid: BoardGrid = initialBoard();
let moveHistory: Move[] = [];
let solving = false;
let currentGen = 1;

const svg = document.getElementById('board') as unknown as SVGSVGElement;
const marbleCountEl = document.getElementById('marble-count')!;
const moveCountEl = document.getElementById('move-count')!;
const confidenceEl = document.getElementById('confidence')!;
const statusEl = document.getElementById('status')!;
const solveBtn = document.getElementById('solve-btn') as HTMLButtonElement;
const resetBtn = document.getElementById('reset-btn') as HTMLButtonElement;
const speedSlider = document.getElementById('speed') as HTMLInputElement;
const genBtns = document.querySelectorAll('.gen-btn');

function updateStats(confidence?: number) {
  marbleCountEl.textContent = String(countMarbles(currentGrid));
  moveCountEl.textContent = String(moveHistory.length);
  confidenceEl.textContent = confidence !== undefined ? `${(confidence * 100).toFixed(0)}%` : '--';
}

function setStatus(text: string) {
  statusEl.textContent = text;
}

function getSpeed(): number {
  return Number(speedSlider.value);
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
    setStatus('Ready');
    solveBtn.disabled = false;
  } catch (e) {
    setStatus('Model not found');
    console.error(e);
  }
}

function reset() {
  solving = false;
  currentGrid = initialBoard();
  moveHistory = [];
  renderBoard(svg, currentGrid);
  updateStats();
  setStatus('Ready');
  solveBtn.disabled = !isModelLoaded();
  solveBtn.textContent = 'Solve';
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

    // Animate
    await new Promise<void>(resolve => {
      animateMove(svg, currentGrid, move, resolve, getSpeed());
    });

    currentGrid = newGrid;
    updateStats(confidence);

    // Small delay between moves
    await new Promise(r => setTimeout(r, 50));
  }

  solving = false;
  solveBtn.textContent = 'Solve';
  resetBtn.disabled = false;
}

// Wire up events
solveBtn.addEventListener('click', solve);
resetBtn.addEventListener('click', reset);
genBtns.forEach(btn => {
  btn.addEventListener('click', () => {
    const gen = Number((btn as HTMLElement).dataset.gen);
    reset();
    selectGen(gen);
  });
});

// Initial render
renderBoard(svg, currentGrid);
updateStats();
selectGen(1);
