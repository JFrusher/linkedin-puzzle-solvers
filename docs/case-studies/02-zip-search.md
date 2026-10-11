# Making Zip's search fast where it was slow

## Problem

Zip is a Hamiltonian path: visit every cell once, passing the numbered checkpoints in order. The A* search
was fine on typical boards (medians of 1–6 ms) but had a long tail. Real Zip boards have few numbers, so the
stress case is a sparse board: 4 checkpoints, no walls. On 30 sparse 8×8 boards the p90 was **5.3 s** and the
worst **18.3 s**.

## Method

Each idea had to beat the previous version on the same 30 boards per size, measured by nodes expanded and time.
Ideas that did not move the numbers were dropped, even when they sounded right.

| Change | Evidence | Kept |
|---|---|---|
| Checkerboard parity | identical node counts: the degree and connectivity checks already imply it | no |
| Checkpoint order: reach the next checkpoint without touching a later one | caught 325 of 622 off-path states on the worst 7×7; max nodes 671 → 346 | yes |
| Unvisited cells connected among themselves, not through the head | caught 25% of off-path states on the worst sparse 8×8; max nodes 98,631 → 54,044 | yes |
| Warnsdorff move order: fewest onward moves first | sparse 8×8 p90 3.6 s → 9.8 ms | yes |
| Tarjan bottlenecks: the end is a cut cell, or a cut cell splits off a part the head cannot enter | caught 55% + 20% of the remaining off-path states; worst 7×7 nodes 32,680 → 13,024 | yes |
| Double forced move | caught under 2% | no |
| Skip repeated states (same head, visited set, next checkpoint) | 58% of popped states were repeats; worst 7×7 1.4 → 0.65 s, 8×8 3.0 → 1.2 s | yes |

## A bug the tests caught

Skipping repeated states breaks answer counting: two answers can reach the same state by different routes,
and the skipped route's answers vanish. The fix rebuilds them by joining each skipped route to every ending
already found from that state. The first version failed a brute-force comparison only now and then: it keyed
states by Python's `id()`, and Python reuses the ids of freed objects, so unrelated states collided. Keeping
each processed state alive fixed it. Over 135 many-answer boards the full answer sets now match a brute-force
search, and that test stays in the suite.

## Result

| Sparse boards | p90 before → after | worst before → after |
|---|---|---|
| 6×6 | 8.6 → 3.4 ms | 65 → 6 ms |
| 7×7 | 100 → 8 ms | 2.4 → 0.65 s |
| 8×8 | **5,275 → 9 ms** | **18.3 → 1.2 s** |

Medians on typical boards stayed within noise (2–5 ms), since each state now costs a little more to check.

**Lesson:** measure each idea in isolation on the hard cases. Two ideas that sound standard (parity and
double forced moves) bought nothing here, and the biggest single win was move ordering, not pruning.
