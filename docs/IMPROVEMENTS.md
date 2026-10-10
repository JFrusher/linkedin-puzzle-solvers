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

## Baseline (current generators, so several answers are possible; to be re-measured on one-answer boards)

| Game | Metric | Value |
|---|---|---|
| Queens 7×7–10×10 | explain steps labelled `Search` | 54% → 72% |
| Tango 6×6 | explain steps labelled `SAT` | 16% |
| Zip 8×8 | latency over 5 seeds | 4 ms → 103 ms (nodes 64 → 2,247) |

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
