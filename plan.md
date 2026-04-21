# Marble Solitaire — AlphaZero-Style Solver

## Context

Build a toy AlphaZero-style model that learns to solve European 37-hole marble solitaire through self-play. The web UI tells the **story of the model learning** — showing 5 generations from random flailing to skilled solving.

**Single-player adaptation**: No opponent. "Self-play" = running MCTS episodes. No sign flip during backup. Value = solution quality (+1 = solved, -1 = stuck).

## Educational Content (Beginner-Friendly)

The web UI includes an **"About" / "How It Works"** section explaining concepts to someone new to AI:

1. **The Game** — Rules of marble solitaire (jump over adjacent marble, remove it, goal = 1 remaining). Why it's hard (billions of possible games, most end stuck).

2. **What is a Neural Network?** — The model is a small "brain" that looks at the board and predicts two things:
   - *Policy*: "Which move should I make?" (probability for each possible jump)
   - *Value*: "How good is this position?" (score from -1 bad to +1 great)

3. **What is MCTS?** — Before each move, the model "imagines" hundreds of possible futures (like a chess player thinking ahead). It combines the neural network's intuition with simulated lookahead to make better decisions.

4. **How does it learn?** — The AlphaZero loop:
   - Play many games using MCTS (self-play)
   - Record what happened: which moves were tried, how each game ended
   - Train the neural network on these experiences
   - Repeat — each generation plays better than the last

5. **What you're seeing** — The 5 generations show the model improving from random (Gen 1) to skilled (Gen 5). Gen 1 has never seen a solitaire board. Gen 5 has played thousands of games.

This text is integrated into the web UI as a collapsible sidebar or modal, not a separate page.

## Cloud API Enhancements (Future)

With a cloud LLM API (e.g., Claude), the app could add:

1. **Move Commentary** — After each move, call the API to explain *why* the model chose that move in natural language ("Jumping from (3,1) clears the left corridor, keeping marbles connected"). Show as a chat-style commentary alongside the board.

2. **Generation Comparison Narrator** — When switching between generations, the API explains what the model learned: "Notice how Gen 3 has learned to keep marbles in the center, while Gen 1 scatters them to the edges where they get stuck."

3. **Interactive Q&A** — Users can ask questions about the board state or the AI concepts, and the API answers contextually.

4. **Photo-to-Board** — Use a vision API to let users photograph a real marble solitaire board and have the model solve their actual position.

5. **Strategy Explainer** — The API analyzes the current board state and explains the strategic considerations in plain English.

These are not in scope for v1 but inform the architecture (keep the solver API clean for future integration).

## Core Concept: Learning Journey UI

Train ~50 iterations, export ONNX checkpoints at 5 key moments:

| Gen | Iteration | Label | Expected behavior |
|-----|-----------|-------|-------------------|
| 1 | 1 | "Random" | Flails, gets stuck with 25+ marbles |
| 2 | 5 | "Novice" | Learns basic patterns, ~15-20 marbles |
| 3 | 15 | "Apprentice" | Decent middle game, ~8-12 marbles |
| 4 | 30 | "Skilled" | Gets close, ~3-5 marbles |
| 5 | 50 | "Expert" | Solves or nearly solves consistently |

The UI shows: generation selector, live stats (marbles remaining, confidence), solve rate badge, and a dramatic visual progression.

## Project Structure

```
marble-solitaire-model/
├── pyproject.toml               # torch, numpy, onnx, onnxruntime, pytest
├── .gitignore
├── src/marble_solitaire/
│   ├── __init__.py
│   ├── board.py                 # Board state, move gen, game logic
│   ├── mcts.py                  # Monte Carlo Tree Search
│   ├── model.py                 # Dual-headed CNN (policy + value)
│   ├── self_play.py             # Episode generation + replay buffer
│   ├── train.py                 # Training loop + checkpoint export
│   ├── inference.py             # Solve with trained model
│   └── export.py                # ONNX export (5 checkpoints)
├── tests/
│   ├── test_board.py
│   ├── test_mcts.py
│   ├── test_model.py
│   ├── test_self_play.py
│   ├── test_train.py
│   ├── test_inference.py
│   └── test_export.py
├── models/                      # .gitignore'd; .pt and .onnx files
│   └── .gitkeep
└── web/
    ├── package.json
    ├── vite.config.ts
    ├── index.html
    └── src/
        ├── main.ts              # App shell, generation selector, controls
        ├── board.ts             # SVG rendering + jump animation
        ├── solver.ts            # ONNX inference (greedy policy, 5 models)
        ├── stats.ts             # Stats overlay (marbles, confidence, solve rate)
        └── style.css
```

## Implementation Steps (TDD: failing test first, then code)

### Step 1: Board Engine (`board.py`)

European 37-hole board as 7x7 grid:
```
. . X X X . .
. X X X X X .
X X X X X X X
X X X O X X X   (O = center empty)
X X X X X X X
. X X X X X .
. . X X X . .
```

- `BoardState`: immutable, 7x7 numpy array (0=empty, 1=marble), separate valid mask
- `get_legal_moves()`: returns list of `(row, col, direction)`
- `apply_move(move) -> BoardState`: returns new state
- Move encoding: `(row, col, dir)` where dir in {UP=0, DOWN=1, LEFT=2, RIGHT=3}
- Network output: `(4, 7, 7)` = 196 logits. Index: `dir * 49 + row * 7 + col`
- `to_tensor() -> (2, 7, 7)`: channel 0 = marbles, channel 1 = valid mask
- `create_legal_move_mask(moves) -> (196,)` binary mask

**Tests** (`test_board.py`): initial state has 36 marbles, center empty, 37 valid positions, 4 initial moves, apply_move correctness, terminal detection, tensor shapes, move index roundtrip.

### Step 2: Neural Network (`model.py`)

Dual-headed CNN:
```
Input: (B, 2, 7, 7)
  → Conv2d(2, 64, 3, pad=1) + BN + ReLU
  → Conv2d(64, 64, 3, pad=1) + BN + ReLU
  → Conv2d(64, 64, 3, pad=1) + BN + ReLU

Policy head: Conv2d(64, 4, 1) → reshape to (B, 196)
Value head:  Conv2d(64, 1, 1) → flatten → FC(49, 64) → ReLU → FC(64, 1) → tanh
```

~75K params. 3 conv layers = 7x7 receptive field (entire board).

**Tests** (`test_model.py`): output shapes, value in [-1,1], deterministic in eval mode, param count.

### Step 3: MCTS (`mcts.py`)

Single-player MCTS (no sign flip):

- **Selection**: PUCT = Q + c_puct * P * sqrt(N_parent) / (1 + N)
- **Expansion**: run network → (policy, value), create children
- **Backup**: propagate value up (same sign)
- **Terminal value**: `1.0 if 1 marble else -1.0 + 2.0/remaining`
  - 1 marble → +1.0, 2 → 0.0, 3 → -0.33, 10 → -0.8

**Tests** (`test_mcts.py`): visits sum correctly, prefers winning move, handles terminal, PUCT formula.

### Step 4: Self-Play (`self_play.py`)

- Run MCTS per move, record (state, visit-count policy, eventual outcome)
- Temperature: τ=1.0 first 15 moves, τ=0.1 after
- `ReplayBuffer`: circular buffer, max 50K examples
- `compute_outcome(remaining)`: smooth reward

**Tests** (`test_self_play.py`): correct shapes, outcome range, policy sums to 1, plausible length.

### Step 5: Training Loop (`train.py`)

```
Loss = CrossEntropy(policy, MCTS_policy) + MSE(value, outcome) + L2_reg

for iteration in range(50):
    1. Generate 50 self-play episodes (50 sims each) → replay buffer
    2. Train 10 epochs (batch=256, Adam lr=1e-3, weight_decay=1e-4)
    3. Log: loss, solve rate on 10 test boards
    4. Save checkpoint at iterations 1, 5, 15, 30, 50
```

Seeded with 1-2 known solutions to bootstrap (helps sparse reward on CPU).

**Reduced params for CPU**: 50 episodes/iter, 50 MCTS sims (not 200). ~3 hours total.

**Tests** (`test_train.py`): training step reduces loss, checkpoint save/load.

### Step 6: Inference (`inference.py`)

- `solve_greedy(network, board)`: pick highest-prob valid move each step
- CLI: `python -m marble_solitaire.inference --model models/gen5.pt`

**Tests** (`test_inference.py`): returns valid moves, all legal.

### Step 7: ONNX Export (`export.py`)

Export 5 checkpoints → 5 ONNX files (~300KB each, ~1.5MB total):
- `gen1_random.onnx`, `gen2_novice.onnx`, `gen3_apprentice.onnx`, `gen4_skilled.onnx`, `gen5_expert.onnx`

These go into `web/public/models/` for Vite to bundle.

**Tests** (`test_export.py`): files created, outputs match PyTorch.

### Step 8: Web UI (`web/`)

Vanilla TypeScript + Vite. SVG board + onnxruntime-web.

**Layout:**
```
┌─────────────────────────────────────────────┐
│  Marble Solitaire AI                        │
│  ┌─────────────────────────────────────────┐│
│  │     Generation: [1] [2] [3] [4] [5]    ││
│  │     "Random"  →  →  →  →  "Expert"     ││
│  └─────────────────────────────────────────┘│
│                                             │
│         ┌───────────────────┐               │
│         │                   │               │
│         │   SVG Board       │   Stats:      │
│         │   (animated)      │   Marbles: 36 │
│         │                   │   Move: 0/35  │
│         │                   │   Confidence  │
│         └───────────────────┘               │
│                                             │
│    [▶ Solve]  [↺ Reset]  Speed: [━━━○━━━]  │
└─────────────────────────────────────────────┘
```

- **Generation selector**: 5 buttons, each loads a different ONNX model
- **Board**: SVG circles, marble jump animation (arc path + fade-out of jumped marble)
- **Stats panel**: live marbles remaining, move count, model confidence (softmax max)
- **Solve**: runs greedy policy step-by-step with animation delay
- **Auto-play all 5**: optional button that runs all 5 generations sequentially for dramatic comparison

### Step 9: Monorepo Integration

**Files to modify:**

1. **`deploy/nginx.conf`** — add:
   ```nginx
   location = /marble-solitaire { return 301 /marble-solitaire/; }
   location /marble-solitaire/ {
       add_header Strict-Transport-Security "max-age=31536000; includeSubDomains; preload" always;
       add_header X-Frame-Options "DENY" always;
       add_header X-Content-Type-Options "nosniff" always;
       add_header Referrer-Policy "strict-origin-when-cross-origin" always;
       add_header Content-Security-Policy "default-src 'self'; script-src 'self' 'wasm-unsafe-eval'; style-src 'self' 'unsafe-inline'; object-src 'none'; base-uri 'self'; frame-ancestors 'none';" always;
       add_header Cross-Origin-Embedder-Policy "unsafe-none" always;
       add_header Cross-Origin-Opener-Policy "same-origin" always;
       add_header Cross-Origin-Resource-Policy "same-site" always;
       add_header Permissions-Policy "camera=(), microphone=(), geolocation=(), payment=()" always;
       add_header X-Permitted-Cross-Domain-Policies "none" always;
       try_files $uri /marble-solitaire/index.html;
   }
   ```

2. **`Dockerfile`** — add web build + copy:
   ```dockerfile
   COPY marble-solitaire-model/web/package*.json ./marble-solitaire-model/web/
   RUN cd marble-solitaire-model/web && npm ci --omit=dev
   COPY marble-solitaire-model/web/ ./marble-solitaire-model/web/
   RUN cd marble-solitaire-model/web && npx vite build
   # copy dist
   RUN mkdir -p /usr/share/nginx/html/marble-solitaire && \
       cp -r /app/marble-solitaire-model/web/dist/. /usr/share/nginx/html/marble-solitaire/
   ```

3. **`index.html`** — add app card for `/marble-solitaire/`

4. **Root `vite.config.ts`** — add to staticApps for local dev

## Training Plan (CPU, ~3 hours)

- 50 iterations, 50 episodes/iter, 50 MCTS sims/move
- Seed replay buffer with 1-2 known European solitaire solutions
- 4x symmetry augmentation (rotational)
- Save PyTorch checkpoints at iterations 1, 5, 15, 30, 50
- Export 5 ONNX files after training completes
- Training runs as: `python -m marble_solitaire.train --iterations 50 --output-dir models/`

## Verification

1. `cd marble-solitaire-model && pytest` — all tests green
2. `python -m marble_solitaire.train` — trains, saves 5 checkpoints
3. `python -m marble_solitaire.export` — produces 5 ONNX files
4. `cd web && npm run dev` — browser UI works, all 5 generations load and animate
5. Gen 1 visibly worse than Gen 5
6. Docker build + nginx serves at `/marble-solitaire/`
