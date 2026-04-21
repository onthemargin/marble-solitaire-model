import * as ort from 'onnxruntime-web';
import { BoardGrid, Move, boardToTensor, getLegalMoves } from './board';

// Move encoding: direction * 49 + row * 7 + col
function moveToIndex(move: Move): number {
  return move.dir * 49 + move.row * 7 + move.col;
}

function indexToMove(idx: number): Move {
  const dir = Math.floor(idx / 49);
  const rem = idx % 49;
  return { row: Math.floor(rem / 7), col: rem % 7, dir };
}

export interface PredictResult {
  move: Move;
  confidence: number;
}

const GEN_MODELS: Record<number, string> = {
  1: 'gen1_clueless',
  2: 'gen2_beginner',
  3: 'gen3_intermediate',
  4: 'gen4_advanced',
  5: 'gen5_master',
};

let currentSession: ort.InferenceSession | null = null;
let currentGen = 0;

export async function loadModel(gen: number): Promise<void> {
  const name = GEN_MODELS[gen];
  if (!name) throw new Error(`Invalid generation: ${gen}`);

  if (currentGen === gen && currentSession) return;

  if (currentSession) {
    currentSession.release();
    currentSession = null;
  }

  const url = `./models/${name}.onnx`;
  currentSession = await ort.InferenceSession.create(url);
  currentGen = gen;
}

export async function predictMove(grid: BoardGrid): Promise<PredictResult | null> {
  if (!currentSession) throw new Error('No model loaded');

  const legalMoves = getLegalMoves(grid);
  if (legalMoves.length === 0) return null;

  const inputData = boardToTensor(grid);
  const inputTensor = new ort.Tensor('float32', inputData, [1, 2, 7, 7]);
  const results = await currentSession.run({ board: inputTensor });

  const policyData = results['policy'].data as Float32Array;

  // Mask illegal moves and find best
  const legalIndices = new Set(legalMoves.map(moveToIndex));

  // Apply softmax only on legal moves
  let maxLogit = -Infinity;
  for (const idx of legalIndices) {
    if (policyData[idx] > maxLogit) maxLogit = policyData[idx];
  }

  let sumExp = 0;
  const probs = new Float32Array(196);
  for (const idx of legalIndices) {
    probs[idx] = Math.exp(policyData[idx] - maxLogit);
    sumExp += probs[idx];
  }
  for (const idx of legalIndices) {
    probs[idx] /= sumExp;
  }

  // Pick highest probability legal move
  let bestIdx = -1;
  let bestProb = -1;
  for (const idx of legalIndices) {
    if (probs[idx] > bestProb) {
      bestProb = probs[idx];
      bestIdx = idx;
    }
  }

  if (bestIdx < 0) return null;

  return {
    move: indexToMove(bestIdx),
    confidence: bestProb,
  };
}

export function isModelLoaded(): boolean {
  return currentSession !== null;
}
