# Game deduction rules: plan

## Goal

Each game gets one rule engine. Rules run until nothing changes, and search starts only when they are stuck.
The same engine feeds the solver (pruning) and the explainer (human steps). There is one path and no copy of
the logic.

## Method

1. **One-answer boards.** The generators in `bench/synthetic.py` currently accept boards with several
   answers. Real LinkedIn boards have exactly one answer. So: generate a board, count its answers (stop
   counting at 2), and retry until there is exactly one.
2. **Baseline.** Record the share of Search/SAT steps, the nodes expanded and the median µs per game.
3. **One rule at a time.** Measure each rule against the baseline, and keep it only if the numbers improve.

## Baseline (one-answer boards, `python -m bench.deduction_report`, 20 seeds per size)

| Game | Size | Metric | Value |
|---|---|---|---|
| Queens | 7x7 | `Search` steps | 18% |
| Queens | 8x8 | `Search` steps | 26% |
| Queens | 9x9 | `Search` steps | 25% |
| Queens | 10x10 | `Search` steps | 18% |
| Tango | 6x6 | `SAT` steps | 5% |
| Zip | 5x5 | nodes median / max · µs median / max | 32 / 55 · 743 / 1,269 |
| Zip | 6x6 | nodes median / max · µs median / max | 44 / 109 · 1,380 / 3,802 |
| Zip | 7x7 | nodes median / max · µs median / max | 116 / 671 · 5,632 / 36,063 |

Before the generators made one-answer boards, Queens showed 54–72% `Search` steps and Tango 16% `SAT`
steps. Most of that came from boards with several answers, not from missing rules. Proving that a Zip
8×8 board has one answer takes about 9 s, because the search must exhaust every path; this is the clearest
sign that Zip pruning is weak.

## Results

### Queens (done)

`Search` placements fell from 18–26% to **0%** over 80 one-answer boards from 7x7 to 10x10. Removing one rule at a
time shows each earns its place:

| Rule set | `Search` placements | Other effect |
|---|---:|---|
| All four rules | 0 / 680 | Try and Fail fires 2 times |
| Without Try and Fail | 1 / 680 | |
| Without Confinement | 0 / 680 | Try and Fail fires 40 times |
| Without Shared Attack | 0 / 680 | Try and Fail fires 106 times |
| Single Choice only | 146 / 680 | |

The 2x1 domino rule is the one-region case of Shared Attack: a region stuck in one row attacks the rest of that
row. **`solve` keeps plain DLX.** Running the rules to completion takes 1.1–2.9 ms against 0.3–0.9 ms for DLX,
and DLX already expands about N nodes, so there is no search left to prune. The rules drive the explanation.

### Zip (done)

Real Zip boards have few numbers, so the stress case is sparse boards: 4 checkpoints, no walls, 30 seeds per
size. Each change was kept only after measuring it on the same boards.

| Change | Evidence | Kept |
|---|---|---|
| Checkerboard parity | identical node counts: degree and connectivity checks already imply it | no |
| Checkpoint order (reach the next checkpoint without touching a later one) | caught 325 of 622 off-path states on the worst 7x7; max nodes 671 → 346 | yes |
| Unvisited cells connected on their own, not through the head | caught 25% of off-path states on the worst sparse 8x8; max nodes 98,631 → 54,044 | yes |
| Warnsdorff move order (fewest onward moves first) | sparse 8x8 p90 3.6 s → 9.8 ms | yes |
| Tarjan bottlenecks (end is a cut cell; a cut cell splits off a part the head cannot enter) | caught 55% + 20% of the remaining off-path states; worst 7x7 nodes 32,680 → 13,024 | yes |
| Double forced move | caught < 2% | no |
| Skip repeated states (same head, visited set, next checkpoint) | 58% of popped states are repeats; halves the tail again | **not yet**: answers that meet at the same state must be stitched back together, or answer counting breaks |

| Sparse boards | p90 before → after | worst before → after |
|---|---|---|
| 6x6 | 8.6 → 3.4 ms | 65 → 6 ms |
| 7x7 | 100 → 7 ms | 2.4 → 1.3 s |
| 8x8 | 5,275 → 9 ms | 18.3 → 2.8 s |

Medians on generated boards are unchanged within noise (about 2–5 ms), since each state now costs a little more.
The remaining tail is deep dead branches that no cheap static check found; repeated-state skipping is the
measured next step. Solver soundness: 0 mismatches against the true path over 90 one-answer boards, and valid
paths on 60 sparse boards.

## Rules by game

### Queens
- **Line-locked region:** when a region's candidates all lie in one row or column, clear the rest of that line.
- **k regions in k lines:** when k regions fit inside k rows (or columns), those lines are theirs, so clear
  every other cell in them. The reverse also holds: k lines whose candidates lie in only k regions.
- **Shared attack (the "X"):** a cell attacked by every remaining candidate of a region, row or column cannot
  hold a queen. This covers 2×2, L and T shapes.
- **Try-and-fail:** place a queen, run the rules, and if any unit is left empty, eliminate that cell.

### Zip
- **Forced edges:** a cell with exactly 2 open sides uses both, unless it is the start or end checkpoint.
- **Dead end:** a cell with 1 open side must be the final checkpoint, otherwise the board fails.
- **Checkerboard parity:** colours alternate along the path. Use this to check colour counts up front and to
  bound the number of steps between consecutive checkpoints.
- **Bottleneck:** an unvisited pocket with one entrance must contain the end.

### Tango (no row-uniqueness rule in the real game)
- **Sign chains:** A×B and B×C imply A=C. Propagate along chains.
- **× pair:** two empty cells joined by × hold one sun and one moon, so feed that into the balance counts early.
- **Window:** with 3 of a kind already in a line, spots that would complete a run of three are excluded.
- **Try-and-fail** for whatever remains.

### Wend (orthogonal moves)
- **Nook:** an open cell with one open neighbour is a word end. The search obeys this implicitly already;
  the gain is the explanation and the choice of start cell.
- **Live nook:** newly created dead ends after each placement mark word ends, which prunes earlier.
- **Corridor:** a cell with 2 open neighbours: the word passes through both of them, or ends at the cell.
- **Cut cell:** each side of an articulation cell must be fillable by the remaining word lengths.

### Crossclimb (5 middle words + locked top/bottom pair)
- **Model change:** the 5 middle words form a ladder; then the top and bottom rows unlock under one shared
  clue and extend the ladder at each end.
- **Fixed letters:** positions that never change along the ladder are shared, which narrows candidate lists.
- **End links:** the top and bottom words must each be one letter away from a ladder end.

### Pinpoint
- **Category kinds:** add checks for word-play categories such as "___ X" compounds, hidden words and
  sound-alikes, alongside the meaning-based score.
- **Generality penalty:** reduce the score of very shallow hypernyms (entity, object, whole).

## Order

1. One-answer generators and a baseline script.
2. Queens rules (largest Search share).
3. Zip rules (latency spikes).
4. Tango rules.
5. Wend rules.
6. Crossclimb model change and rules.
7. Pinpoint category kinds (needs real examples).
