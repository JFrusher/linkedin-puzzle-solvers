# LinkedIn Puzzle Solvers Engine

[![CI](https://github.com/JFrusher/linkedin-puzzle-solvers/actions/workflows/ci.yml/badge.svg)](https://github.com/JFrusher/linkedin-puzzle-solvers/actions/workflows/ci.yml)
[![Benchmark](https://github.com/JFrusher/linkedin-puzzle-solvers/actions/workflows/benchmark.yml/badge.svg)](https://github.com/JFrusher/linkedin-puzzle-solvers/actions/workflows/benchmark.yml)
![Python](https://img.shields.io/badge/python-3.11-blue)
![OpenCV](https://img.shields.io/badge/vision-OpenCV%20%7C%20EasyOCR-green)
![Solvers](https://img.shields.io/badge/solvers-DLX%20%7C%20A*%20%7C%20SAT%20%7C%20SBERT%20%7C%20Trie-purple)

A Python monorepo of computer-vision parsers and solvers for LinkedIn's eight daily puzzles. It turns a
screenshot into a structured board, solves it and explains every deduction. It also draws the answer back
onto the original image.

| Game | Vision | Solver |
|---|---|---|
| **Queens** (Crowns) | LAB colour sampling + K-Means regions | Knuth's Algorithm X on Dancing Links |
| **Zip** (Pathfinder) | Hough circles + EasyOCR digits, wall-thickness probes | A* with connectivity and degree pruning |
| **Tango** (Sun & Moon) | Symbol colour (orange sun, blue moon) + ink at each edge sign's centre (× vs =) | CNF reduction solved by MiniSat (PySAT) |
| **Pinpoint** (Categories) | Text stream | Sentence-BERT moving centroid × WordNet hypernyms + word-play checks (join word, many senses) |
| **Crossclimb** (Word Ladder) | Letter-box rows + EasyOCR clues, or typed in | SBERT clue answering + Hamming-1 ladder search + shared-clue end pair |
| **Wend** (Word Partitioning) | Per-tile EasyOCR on tight crops, grey wall tiles, slot rows (same or separate screenshot), or typed in | Trie-guided DFS tiling over WordNet words and their inflections + flood-fill island pruning |
| **Mini Sudoku** (6×6) | Ink test + per-cell EasyOCR digits | CNF exactly-one encoding solved by MiniSat (PySAT) |
| **Patches** (Rectangles) | Coloured badges: a '+' badge holds the area (EasyOCR), a solid icon's aspect gives the shape | Exact cover on Dancing Links |

> 📸 **Demo placeholder:** `docs/demo.gif` — screenshot → overlay round-trip in the browser extension.
>
Solved overlays on real LinkedIn screenshots (`tests/fixtures/real/`):

| Queens | Zip | Tango |
|:---:|:---:|:---:|
| ![Queens](docs/queens.png) | ![Zip](docs/zip.png) | ![Tango](docs/tango.png) |
| **Mini Sudoku** | **Patches** | **Wend** |
| ![Mini Sudoku](docs/mini-sudoku.png) | ![Patches](docs/patches.png) | ![Wend](docs/wend.png) |

---

## Quick start

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu   # CPU build, ~2 GB smaller
pip install -r requirements.txt
python -m nltk.downloader wordnet

python main.py solve --game queens --image board.png        # writes board_solved.png
python main.py solve --game zip --image zip.png --json      # machine-readable result
python main.py solve --game wend --image wend.png --slots slots.png   # word slots in their own screenshot
python main.py solve --game wend --text examples/wend.txt   # or type the board in (Wend, Crossclimb)
python main.py stream --game pinpoint                       # type clues one at a time
python main.py benchmark --game all                         # latency / memory / accuracy
python main.py serve                                        # REST backend for the extension
streamlit run web_app/app.py                                # interactive playground
```

The first OCR or embedding call downloads the EasyOCR weights (GitHub) and `all-MiniLM-L6-v2` (Hugging Face).
The Docker image bakes both in:

```bash
docker build -t puzzle-solvers .
docker run -p 8765:8765 puzzle-solvers                       # serve
docker run -v "$PWD:/data" puzzle-solvers solve --game tango --image /data/tango.png
```

---

## Architecture

```text
                   ┌──────────────────────────────────────────────────────────────┐
  screenshot ───►  │ core/vision_base.py  BaseVisionParser                        │
                   │  adaptiveThreshold → findContours → largest convex quad      │
                   │  getPerspectiveTransform → 720×720 deskewed grid             │
                   │  edge-profile × line-comb correlation → N                    │
                   │  extract_cells(N) → trimmed cell crops                       │
                   └───────────────┬──────────────────────────────────────────────┘
                                   │ game parser (solvers/<game>/vision.py)
                                   ▼
                          typed, validated Board dataclass
                                   │
                   ┌───────────────▼──────────────────────────────────────────────┐
                   │ core/solver_base.py  BaseSolver.run()                        │
                   │  profile(): perf_counter_ns (µs) + tracemalloc (peak KiB)    │
                   │  solve(board) → solution | UnsolvableError                   │
                   │  nodes_expanded counter                                      │
                   └───────┬───────────────────────────────┬──────────────────────┘
                           │                               │
             ┌─────────────▼───────────┐       ┌───────────▼────────────────────┐
             │ core/explainer.py       │       │ core/visualizer.py             │
             │ "Step 2 [Single Choice]"│       │ crowns, suns/moons, ribbons    │
             │ → JSON / Markdown       │       │ BGRA layer → inverse homography│
             └─────────────────────────┘       └───────────┬────────────────────┘
                                                           ▼
        CLI (main.py) · Streamlit (web_app/) · REST (main.py serve) ◄── extension/
```

### Browser extension flow

```text
 toolbar click ─► background.js: captureVisibleTab (activeTab)
              ─► POST 127.0.0.1:8765/solve {game, image}
              ◄─ {solution, steps, profile, overlay: BGRA PNG}
              ─► content.js: full-viewport <img>, pointer-events: none  (Esc hides)
```

The extension does not rely on LinkedIn's DOM class names. It runs the same vision pipeline on a screenshot of the viewport.

---

## Algorithms and complexity

| Game | Formulation | Worst case | In practice |
|---|---|---|---|
| Queens | Exact cover: 3N primary items (row, column, region) and (N−1)² secondary 2×2 windows encoding non-adjacency | O(N!) branches (NP-complete) | MRV column choice; ≤ N + a few nodes on generated boards |
| Zip | Hamiltonian path through ordered checkpoints; A* over (head, visited bitmask, next checkpoint) | O(4^{N²}) states (NP-complete) | f = g + max(unvisited, Manhattan chain) is exact on feasible states. Flood-fill and degree-2 pruning cut dead ends |
| Tango | SAT: N² vars; sequential-counter cardinality for N/2 per line; (¬a∨¬b∨¬c)∧(a∨b∨c) per triple; ⇔ / ⊕ for signs | O(2^{N²}) | MiniSat CDCL, tens of decisions |
| Pinpoint | C = (1/K) Σ Embed(clueᵢ); score = cos(C, Embed(h)) × coverage(h) over hypernyms h | O(K · H · d) | H ≈ hundreds of hypernyms, d = 384 |
| Crossclimb | Top-25 gloss-similar words per clue; branch-and-bound best-score path in the Hamming-1 graph; then the top/bottom pair one letter from each end that best fits the shared clue | O(K! · 25^K) | Bounded by best-remaining-score; sparse graph; end pairs are tens of candidates |
| Mini Sudoku | SAT: 216 vars x(r,c,d); pairwise exactly-one per cell and per digit in every row, column and 2×3 box | O(6^{36}) | Givens propagate to the answer: MiniSat makes ~2 decisions |
| Patches | Exact cover: N² cell items + one item per clue; rows are rectangles that fit the clue's area and/or shape and hold no other clue | O(rectangles^{clues}) | MRV column choice; tens of nodes |
| Wend | Trie DFS enumerates word paths; exact tiling over tiles with MRV; islands must be subset-sum fillable | O(placements^{words}) | Trie prunes dead prefixes; flood-fill rejects untileable islands |

### Measured performance

Below is the median of 25 solves without memory tracing, plus one traced run for peak memory. It was run on a
shared cloud container with CPython 3.13. Vision ✅ means the parser recovered the hand-checked board: from a
perspective-skewed synthetic render, or from a real LinkedIn screenshot. Every board has exactly one answer,
like the real game.

| Game | Case | Vision | Solved | Median latency (µs) | Peak memory (KiB) | Nodes |
|---|---|:---:|:---:|---:|---:|---:|
| queens | 6x6 | ✅ | ✅ | 151 | 28.5 | 6 |
| queens | 8x8 | ✅ | ✅ | 282 | 50.2 | 9 |
| queens | 10x10 | ✅ | ✅ | 437 | 78.4 | 10 |
| queens | 12x12 | ✅ | ✅ | 746 | 113.0 | 12 |
| queens | real screenshot (9x9) | ✅ | ✅ | 1,074 | 63.6 | 81 |
| zip | 5x5 | ✅ | ✅ | 1,360 | 19.9 | 31 |
| zip | 6x6 | ✅ | ✅ | 1,520 | 25.9 | 36 |
| zip | 7x7 | ✅ | ✅ | 5,923 | 40.0 | 81 |
| zip | real screenshot (7x7) | ✅ | ✅ | 4,036 | 32.4 | 62 |
| tango | 6x6 seed 1 | ✅ | ✅ | 358 | 75.2 | 8 |
| tango | 6x6 seed 2 | ✅ | ✅ | 361 | 76.1 | 9 |
| tango | 6x6 seed 3 | ✅ | ✅ | 337 | 75.4 | 2 |
| tango | real screenshot (6x6) | ✅ | ✅ | 368 | 75.5 | 8 |
| wend | 5x5 sample | ✅ | ✅ | 832 | 31.3 | 368 |
| wend | real screenshot (8x8, 5 words) | ✅ | ✅ | 1,612 | 34.1 | 622 |
| mini-sudoku | 6x6 seed 1 | ✅ | ✅ | 908 | 328.2 | 2 |
| mini-sudoku | 6x6 seed 2 | ✅ | ✅ | 757 | 328.3 | 2 |
| mini-sudoku | 6x6 seed 3 | ✅ | ✅ | 758 | 328.3 | 2 |
| mini-sudoku | real screenshot | ✅ | ✅ | 759 | 328.3 | 2 |
| patches | 5x5 | ✅ | ✅ | 385 | 18.7 | 11 |
| patches | 6x6 | ✅ | ✅ | 1,062 | 32.2 | 14 |
| patches | 7x7 | ✅ | ✅ | 1,492 | 43.8 | 21 |
| patches | real screenshot (8x8) | ✅ | ✅ | 2,052 | 72.6 | 16 |

Queens meets the < 1 ms target up to 12×12 on this run (12×12: 746 µs); the real 9×9 board takes 1.1 ms because its huge yellow region leaves DLX more branching (81 nodes). Zip on
sparse boards (few numbers, no walls) is the hardest case; its numbers are in `docs/IMPROVEMENTS.md`. Pinpoint and Crossclimb latencies come from the CI benchmark (they need the
SBERT model). The **Benchmark** workflow re-runs the whole suite daily and publishes the table as a job summary
and an artifact.

---

## Explainable output

Every run returns a deduction log, rendered as Markdown in the CLI or as JSON over REST:

```text
Queens 8×8 (seed 3)
| 4  | Single Choice | (5,5) | Cell (5, 5) must be a Queen because Region 6 has no other valid slot   |
Tango 6×6 (seed 4)
| 9  | No Three      | (4,3) | Cell (4, 3) must be sun because (2, 3) and (3, 3) are both moon        |
| 10 | SAT           | (0,0) | Cell (0, 0) must be sun because the SAT model fixes it; no single rule applies |
| 11 | Relation      | (1,0) | Cell (1, 0) must be sun because of the '=' sign with (0, 0)            |
| 14 | Balance       | (2,4) | Cell (2, 4) must be sun because its line already holds 3 moons         |
```

Queens, Tango, Wend, Mini Sudoku and Patches replay the solution with human rules (single choice, relations,
no-three, balance, naked and hidden singles, only fit, only cover, try and fail). They label a step `Search` or `SAT`
only when no single rule forces the move.

---

## Repository layout

```text
core/          vision_base · solver_base · explainer · visualizer · dlx · nlp (SBERT + WordNet)
solvers/       queens · zip · tango · pinpoint · crossclimb · wend · sudoku · patches  (+ registry in __init__)
bench/         benchmark_runner · synthetic (seeded generators + skewed renderers)
tests/         one suite per game + core
web_app/       Streamlit playground (Raw · Contours · HSV Masks · Graph Nodes tabs)
extension/     Manifest V3: background.js (capture + REST) · content.js (overlay)
main.py        CLI: solve · stream · benchmark · serve
```

---

## Scope and limitations

- **Calibrated on one real screenshot per game** (`tests/fixtures/real/`, boards in `bench/real.py`) and on
  seeded synthetic renders that copy the real look. Every threshold quotes the real and synthetic values it
  separates. Crossclimb and Pinpoint have no real capture yet, so the Crossclimb layout is still a guess.
- **Crossclimb** assumes each clue is printed to the left of its row of letter boxes. It answers clues from
  WordNet glosses, so trivia outside WordNet's vocabulary is out of reach.
- **Wend** words use orthogonal adjacency. Inflections (DOWNLOADED, CARRIED) are generated by English
  spelling rules and kept only when WordNet maps them back to a lemma, so a few odd forms of mass nouns
  ("wealths") get in. Without the word lengths the solver finds *a* tiling, not necessarily the intended one:
  pass `--slots` when the slot rows are not in the board screenshot, or type the board with its lengths.
  The REST server takes one image.
- **Typed boards** (`--text`, or "Type the board" in the playground) skip vision for Wend and Crossclimb.
  Wend: one row of letters per line, `#` for a wall, then the word lengths. Crossclimb: the word length, the
  shared top/bottom clue, then the middle clues. See `examples/`.
- **Zip** uses A* with backtracking order and pruning. It does not use a bidirectional search.
