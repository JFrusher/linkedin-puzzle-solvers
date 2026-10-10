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
| Skip repeated states (same head, visited set, next checkpoint) | 58% of popped states are repeats; worst 7x7 1.4 → 0.65 s, 8x8 3.0 → 1.2 s | yes: answers that meet at a skipped state are rebuilt by splicing; checked against brute force |

| Sparse boards | p90 before → after | worst before → after |
|---|---|---|
| 6x6 | 8.6 → 3.4 ms | 65 → 6 ms |
| 7x7 | 100 → 8 ms | 2.4 → 0.65 s |
| 8x8 | 5,275 → 9 ms | 18.3 → 1.2 s |

Medians on generated boards are unchanged within noise (about 2–5 ms), since each state now costs a little more.
Repeated-state skipping needs care when counting answers. Two answers can reach the same state by different
routes, so when a repeat is skipped, its answers are rebuilt by joining its route to every ending already
found from that state. Over 135 many-answer boards, the full answer sets matched the solver without
skipping. The first version failed this check only now and then: it keyed states by `id()`, and Python
reused the ids of freed states. Keeping each processed state alive fixed it. A brute-force test now
guards it. Solver soundness: 0 mismatches against the true path over 90 one-answer boards, and valid
paths on 60 sparse boards.

### Tango (done)

`SAT` steps over 40 one-answer boards (the 20-seed baseline was 5%):

| Rules | `SAT` steps | Try and Fail steps |
|---|---:|---:|
| Relation, No Three, Balance | 72 | 0 |
| + Sign Balance | 30 | 0 |
| + Equal Pair Neighbour | 15 | 0 |
| + Try and Fail | **0** | 18 |

The soundness test (no deduction may contradict the true grid) caught two overlap bugs in Sign Balance
before they shipped:

- a chain a×b×c holds two of one symbol, not one of each per pair;
- an = pair that shares a cell with a counted × pair needs only one more slot.

After the fixes there were 0 wrong deductions over 1,000 boards, and the rules alone filled every cell.
`solve` keeps SAT (under 1 ms).

### Wend (done)

Solving is already fast (5x5 worst 3.9 ms, 6x6 worst 6.2 ms, with or without word lengths), so these rules feed
the explanation only. The new generator writes random dictionary words along random snakes and keeps only
one-answer boards. About 40% of raw boards have one answer, since WordNet is far larger than the game's
dictionary.

| Rules | `Search` words (20 boards, 5x5 / 6x6) |
|---|---|
| Replay in solution order (before) | 100% |
| Only Word + Nook | 27% / 27% |
| + Island + Try and Fail | 0% / 0% (398 log steps at 5x5) |
| + Try and Fail, no Island | **0% / 0%** (299 log steps at 5x5): Island was redundant, so it was removed |

**Nook (your rule):** a tile with one open neighbour can only be a word's first or last letter. The search
already obeys this implicitly, because no path can pass through such a tile. In the log it marks the step
where that tile's only fitting word is placed. Corridor and cut-cell pruning were skipped, because there is
no speed problem to fix.

### Crossclimb (done)

The real board is a 7-word ladder. Five middle words have their own clues and must be ordered. The top and
bottom words unlock last and share one clue: together they form a pair or phrase, read in either order.
Puzzles found online confirm this: TEAM … WORK (teamwork), LIMA … KILO (NATO letters), TOWN … HALL,
FLIP … FLOP.

The solver orders the middle ladder as before, then scores every pair of words one letter from each end of
the ladder. Each pair is described in both reading orders: a WordNet compound with its gloss ("teamwork:
cooperative work done by a team"), or else the plain phrase. The description closest to the shared clue
wins. TEAM/WORK has only 8 × 9 candidate pairs, so this step is cheap. The sample board is now the
5 October 2026 puzzle, TEAM-REAM-ROAM-FOAM-FORM-WORM-WORK. The middle ladder and the end pair both rely
on SBERT, which runs in CI; the pair selection logic is also tested locally with a stand-in encoder.

### Mini Sudoku and Patches (new games)

Both reuse existing engines. Mini Sudoku is a SAT encoding like Tango; Patches is exact cover like Queens,
so Dancing Links moved from the Queens solver to `core/dlx.py`. Their generators make one-answer boards
(Sudoku removes givens while the answer stays unique; Patches retries tilings until one is unique).

| Game | Size | Metric | Value |
|---|---|---|---|
| Mini Sudoku | 6x6 | `SAT` steps | 0% (Naked Single 507, Hidden Single 12; 9–11 givens) |
| Patches | 5x5 | `Search` patches | 0% (Only Fit 134, Only Cover 51, Try and Fail 30) |
| Patches | 6x6 | `Search` patches | 0% (Only Fit 198, Only Cover 62, Try and Fail 61) |
| Patches | 7x7 | `Search` patches | 0% (Only Fit 228, Only Cover 115, Try and Fail 44) |

Patches boards follow the real game: a clue gives an area or a shape, not both. Shape-only clues first left
5–7% of patches to `Search`. Try and Fail, as in Queens, Tango and Wend, rules out a rectangle when Only Fit
and Only Cover steps after it leave a clue with no rectangle or a cell nothing can cover. That brought
`Search` to 0%; the real board needs two Try and Fail steps. No ruled-out rectangle was ever part of the
answer over 60 boards.

## Real screenshots

One real capture per game is in `tests/fixtures/real/`, with the board read by hand in `bench/real.py`. Each
hand-read board solves to exactly one answer, which checks the transcription. Running every parser against
them first gave:

| Game | First run | Cause (measured) | Fix |
|---|---|---|---|
| Queens | ✅ 9 regions | — | — |
| Zip | one extra wall | the "1" disc reaches the wall probe: 0.051 of a cell vs threshold 0.05; real walls 0.194, synthetic 0.097 | threshold 0.075 |
| Mini Sudoku | missed all three 1s | a "1" is 0.016 ink vs threshold 0.02; empty cells 0 | threshold 0.005 |
| Tango | every symbol swapped, all signs missed | grey templates of our icons fit LinkedIn's badly; sign ink is grey 117+, test was < 100 | symbols by colour (blue − red: moon +145, sun −204); signs: ink < 160, and an "×" has ink at its centre (1.0) where "=" has a gap (0.0) |
| Patches | crash | the layout was a guess: clues are coloured badges, an area or a shape | parse badges: saturation > 50 finds all 12; filled share 0.98+ = plain shape, 0.79–0.85 = "+" badge; white holes = digits |
| Wend | crash | walls are light grey (178) not dark; loose crops read U as "PU" (36/40 tiles); WordNet has DOWNLOAD but not DOWNLOADED | median grey < 215; tight crops (40/40); add inflections; slots may come as their own screenshot |

The synthetic renders were moved toward the real look where they differed (the "=" sign's bar gap, Patches
badges, white digit weight), so the synthetic tests keep checking the same features. All six games now parse
their real screenshot exactly and solve it; `tests/test_real.py` keeps it that way.

### Pinpoint (done)

`solvers/pinpoint/wordplay.py` adds three WordNet-only checks; the measurements are on the real puzzles
listed below plus my earlier samples as controls.

| Check | Evidence |
|---|---|
| Join word | "breaker" 3/5; "whale" 4/5; no join above 1/5 for chess or instruments |
| Subtype filter (red oak is an oak) | without it, "words after *red*" covered 5/5 trees, a false answer |
| Word play ranks first only if it covers more clues than the best specific category (WordNet depth ≥ 7) | breaker 3 vs 1 → join wins; trees "family" 4 vs tree 5 → meaning wins; chess 1 vs 5 |
| Clue + class word (blue → blue whale → whale) | whale coverage 5/5 instead of 3/5 |
| Many senses | invented "point" set: 3/5 (no real clue list was found); every control ≤ 2 |

Building the indexes first took 16 s, almost all of it NLTK parsing every WordNet synset. Counting meanings
from the lemma index and parsing only words with 5+ meanings brought the build to 6.8 s, including
WordNet's own load. "___ ten" style phrases remain out of reach without a phrase list.

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

Real puzzles from fan answer sites (unofficial; check against the game before relying on them):

| Date | Clues | Category | Kind |
|---|---|---|---|
| 10 Oct 2026 (#893) | Hang, Perfect, Top, Count to, Nine times out of | not published yet; every clue fits "___ ten" | fill-in-the-blank phrase |
| 9 Oct 2026 (#892) | Blue, Right, Bowhead, Humpback, Orca | Types of whales | members of a class |
| 8 Oct 2026 (#891) | Code, Circuit, Ice, Deal, Tie | Words before "breaker" | compound word |
| 3 Oct 2026 | (clues not found) | Words after "common" | compound / phrase |
| 30 Sep 2026 | (clues not found) | Types of rings | members of a class |
| 27 Sep 2026 | (clues not found) | Words before "share" | compound word |
| 26 Sep 2026 | (clues not found) | Different definitions of "point" | one word, many senses |
| 25 Sep 2026 | (clues not found) | Words after "mixed" | compound / phrase |
| 24 Sep 2026 | (clues not found) | Things associated with steam | association |
| 23 Sep 2026 | (clues not found) | Things shaped like triangles | shared property |
| 21–22 Sep 2026 | (clues not found) | Common national demonyms; Famous philosophers | members of a class |

About half the categories are word play: a hidden word joins every clue into a compound or phrase
("code breaker", "hang ten"). The current engine only scores meaning (hypernyms + embeddings), so it
cannot find these. Three checks would cover what was found:

- **Join word:** find a word X such that "clue X", "X clue" or "clueX" is a WordNet compound for most
  clues. Checked: WordNet covers 3 of 5 "breaker" clues (circuit breaker, icebreaker, tiebreaker; not
  code breaker or deal breaker), so "most clues" is enough to win there. It has none of the "ten" phrases
  (hang ten, perfect ten, top ten), so those need a phrase list WordNet lacks.
- **Partial-name class:** "Blue", "Right" and "Bowhead" are whale names only with "whale" added, so try
  "clue + candidate" as a lemma as well as hypernyms. Checked: blue whale, right whale, bowhead whale and
  humpback whale are all WordNet lemmas, and orca is one on its own, so this covers 5 of 5.
- **Many senses:** when every clue matches a different WordNet sense of one word, rank that word as the
  category. "point" has 40 WordNet senses, so this check must want most clues to land on distinct senses.

## Order

1. One-answer generators and a baseline script.
2. Queens rules (largest Search share).
3. Zip rules (latency spikes).
4. Tango rules.
5. Wend rules.
6. Crossclimb model change and rules.
7. Pinpoint category kinds (needs real examples).
