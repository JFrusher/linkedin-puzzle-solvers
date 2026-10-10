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
| **Tango** (Sun & Moon) | Template matching (cells and edge signs) | CNF reduction solved by MiniSat (PySAT) |
| **Pinpoint** (Categories) | Text stream | Sentence-BERT moving centroid × WordNet hypernyms + word-play checks (join word, many senses) |
| **Crossclimb** (Word Ladder) | Letter-box rows + EasyOCR clues | SBERT clue answering + Hamming-1 ladder search + shared-clue end pair |
| **Wend** (Word Partitioning) | Per-tile EasyOCR, wall tiles, slot rows | Trie-guided DFS tiling + flood-fill island pruning |
| **Mini Sudoku** (6×6) | Ink test + per-cell EasyOCR digits | CNF exactly-one encoding solved by MiniSat (PySAT) |
| **Patches** (Rectangles) | EasyOCR area digits + icon aspect (square / tall / wide) | Exact cover on Dancing Links |

> 📸 **Demo placeholder:** `docs/demo.gif` — screenshot → overlay round-trip in the browser extension.
>
> 🖼️ **Overlay placeholders:** `docs/queens.png`, `docs/zip.png`, `docs/tango.png`, `docs/wend.png`, `docs/mini-sudoku.png`, `docs/patches.png`

---

## Quick start

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu   # CPU build, ~2 GB smaller
pip install -r requirements.txt
python -m nltk.downloader wordnet

python main.py solve --game queens --image board.png        # writes board_solved.png
python main.py solve --game zip --image zip.png --json      # machine-readable result
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
| Patches | Exact cover: N² cell items + one item per clue; rows are rectangles of the clue's area and shape holding no other clue | O(rectangles^{clues}) | MRV column choice; tens of nodes |
| Wend | Trie DFS enumerates word paths; exact tiling over tiles with MRV; islands must be subset-sum fillable | O(placements^{words}) | Trie prunes dead prefixes; flood-fill rejects untileable islands |

### Measured performance

Below is the median of 25 solves without memory tracing, plus one traced run for peak memory. It was run on a
shared cloud container with CPython 3.13 on synthetic boards. Vision ✅ means the parser recovered the
ground-truth board from a perspective-skewed render. Every board has exactly one answer, like the real game.

| Game | Case | Vision | Solved | Median latency (µs) | Peak memory (KiB) | Nodes |
|---|---|:---:|:---:|---:|---:|---:|
| queens | 6x6 | ✅ | ✅ | 181 | 28.5 | 6 |
| queens | 8x8 | ✅ | ✅ | 310 | 50.2 | 9 |
| queens | 10x10 | ✅ | ✅ | 598 | 78.4 | 10 |
| queens | 12x12 | ✅ | ✅ | 955 | 113.0 | 12 |
| zip | 5x5 | ✅ | ✅ | 1,275 | 19.9 | 31 |
| zip | 6x6 | ✅ | ✅ | 1,779 | 25.9 | 36 |
| zip | 7x7 | ✅ | ✅ | 6,896 | 40.0 | 81 |
| tango | 6x6 seed 1 | ✅ | ✅ | 508 | 75.2 | 8 |
| tango | 6x6 seed 2 | ✅ | ✅ | 498 | 76.1 | 9 |
| tango | 6x6 seed 3 | ✅ | ✅ | 438 | 75.4 | 2 |
| wend | 5x5 sample | ✅ | ✅ | 1,148 | 31.0 | 363 |
| mini-sudoku | 6x6 seed 1 | ✅ | ✅ | 1,390 | 328.2 | 2 |
| mini-sudoku | 6x6 seed 2 | ✅ | ✅ | 2,040 | 328.3 | 2 |
| mini-sudoku | 6x6 seed 3 | ✅ | ✅ | 1,405 | 328.3 | 2 |
| patches | 5x5 | ✅ | ✅ | 199 | 13.3 | 8 |
| patches | 6x6 | ✅ | ✅ | 506 | 23.1 | 17 |
| patches | 7x7 | ✅ | ✅ | 722 | 32.0 | 22 |

Queens meets the < 1 ms target up to 12×12 on this run (12×12: 955 µs); DLX node allocation dominates. Zip on
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

Queens, Tango, Mini Sudoku and Patches replay the solution with human rules (single choice, relations,
no-three, balance, naked and hidden singles, only fit and only cover). They label a step `Search` or `SAT`
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

- **Calibrated on synthetic boards.** Thresholds and templates are tuned against seeded, perspective-skewed
  renders in `bench/synthetic.py`. Real LinkedIn screenshots use different icons, fonts and colours, so expect
  to recalibrate the Tango templates, the Zip disc and wall thresholds, and the Crossclimb layout once real
  captures are added to the test set.
- **Crossclimb** assumes each clue is printed to the left of its row of letter boxes. It answers clues from
  WordNet glosses, so trivia outside WordNet's vocabulary is out of reach.
- **Wend** words use orthogonal adjacency.
- **Patches** expects the area in the top half of a clue cell and a rectangle icon (square, tall or wide)
  below it, with no icon meaning any shape. That layout is from the synthetic renders; check it against a
  real capture.
- **Zip** uses A* with backtracking order and pruning. It does not use a bidirectional search.
