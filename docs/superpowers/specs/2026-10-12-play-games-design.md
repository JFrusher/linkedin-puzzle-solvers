# Play: generate and play the grid games locally

**Date:** 2026-10-12 · **Status:** approved design, awaiting spec review

## Goal

A sister app to `web_app/` where you play freshly generated puzzles for the six grid games (Queens, Zip,
Tango, Mini Sudoku, Patches, Wend) at a chosen Easy / Medium / Hard level, in the browser, on your own machine.

Success means you can start `python main.py play`, pick a game and a level, and get a new board straight away.
You can then solve it with LinkedIn-style mouse controls, see rule breaks as you make them, undo, ask for a hint,
and get your time recorded against your personal best.

## Decisions (from the brainstorm)

| Topic | Decision |
|---|---|
| Audience | Just you, run locally. No accounts or hosting, and no anti-cheat. |
| Platform | A static HTML/CSS/JS page plus a small Python API, in a new `play/` folder |
| Games | Queens, Zip, Tango, Mini Sudoku, Patches, Wend. Not Crossclimb or Pinpoint: no clue source. |
| Difficulty | Easy / Medium / Hard, targeted with `core.difficulty.rate`. Zip uses size and clue count. |
| Features | Hints, undo / redo / clear, timer, personal bests. No reveal-answer, no daily puzzle. |
| Mistakes | Rule breaks highlighted live. The board is compared with the answer only for hints and the win check. |
| Speed | A background pool keeps boards ready, so New is instant. |
| Controls | LinkedIn-style mouse controls. No keyboard controls beyond typing Sudoku digits. |
| Look | Close to LinkedIn's colours and shapes. |
| Architecture | Option A: the browser receives the board, answer and deduction log, and does all play logic itself. |

## Architecture

```
python main.py play ──► play/server.py (ThreadingHTTPServer)
                           ├── GET /                     → play/static/index.html (+ css, js modules)
                           └── GET /api/puzzle?game&level → play/pool.py ──► play/levels.py ──► bench/synthetic.py
                                                                                              └► solvers/*  (solve + explain)
                                                                                              └► core/difficulty.rate
browser: play/static/app.js (shell) + play/static/games/<game>.js (one module per game)
```

Generation stays in Python, so the existing generators, solvers and explainers are reused unchanged. Play logic
lives in the browser, so clicks get an instant response.

### `play/levels.py`

`LEVELS: dict[str, dict[str, LevelSpec]]` maps each game to its three levels. A `LevelSpec` holds:

- `make(seed) -> board`: a call into the existing `bench.synthetic` generator, with the settings for that level.
- `accept: frozenset[str]`: the rating bands that count as this level (`Easy` = Trivial or Easy,
  `Medium` = Medium, `Hard` = Hard or Expert). It is empty for Zip, which has no rating, so any board is accepted.

The starting settings are below. They get tuned during implementation by sampling ratings, as was done in the brainstorm.

| Game | Easy | Medium | Hard |
|---|---|---|---|
| Queens | 6×6 | 7×7 | 9×9 |
| Tango | 6×6, 6 givens, 6 signs | 6×6, 4 givens, 4 signs | 6×6, 2 givens, 2 signs |
| Mini Sudoku | 6×6 (accept Easy) | 6×6 (accept Medium) | 6×6 (accept Hard or above) |
| Patches | 5×5 | 6×6 | 7×7 |
| Wend | 4×4 | 5×5 | 6×6 |
| Zip | 5×5, 8 checkpoints, 0 walls | 6×6, 6 checkpoints, 2 walls | 7×7, 4 checkpoints, 4 walls |

At default settings the brainstorm sample never produced Hard for Queens 7×7 or Mini Sudoku, and never Easy for
Patches 6×6. That is why sizes vary by level and why the pool has a fallback (below).

### `play/pool.py`

- One daemon thread fills a store of up to **3** boards for each (game, level). Seeds come from `random.SystemRandom`.
- Each candidate is solved, explained and rated, then filed under the level its *actual* rating falls in. So a
  board made while aiming for Hard but rated Medium tops up Medium instead of being thrown away.
- If a (game, level) gets nothing after **200** tries or **60 s**, the pool serves the closest-rated board it has.
  The puzzle carries its true `rating` string, and the UI shows it.
- `take(game, level)` returns a ready board immediately. If that level is empty it generates one on the calling
  thread (the first request at startup may wait; Wend is about 4 s).
- `take` wakes the filler thread so the store refills.

### `play/server.py`

A `ThreadingHTTPServer` built like `main.py`'s `SolveHandler`: JSON responses and a socket timeout.

- `GET /` and `GET /static/...`: files from `play/static/`. Paths are resolved and must stay inside that folder;
  anything else gets 404.
- `GET /api/puzzle?game=<name>&level=easy|medium|hard` returns 200 with:
  ```json
  {"game": "queens", "level": "hard", "rating": "Hard: Try and Fail ×2",
   "board": {...}, "solution": ..., "steps": [{"index": 1, "rule": "...", "message": "...", "cell": [r, c]}]}
  ```
  `board` and `solution` use `main.to_jsonable` on the existing dataclasses. An unknown game or level gets 400 with
  `{"error": ...}`.
- `main.py play [--port 8770] [--no-browser]` starts the pool and the server, then opens the browser.

### Front end: `play/static/`

Plain HTML, CSS and ES modules, with no build step and no npm packages.

- `index.html` and `style.css`: the game picker, level picker, New button, timer, undo / redo / clear, Hint button,
  hint panel and win dialog. Colours follow LinkedIn's (region palette, sun / moon, badge colours).
- `app.js` (the shell) fetches the puzzle and owns the move history: each move is one `{before, after}` change to the
  player state, so undo and redo are generic. It runs the timer, which starts on the first move and stops on a
  win. It stores personal bests in `localStorage` under `play:best:<game>:<level>`, with every read and write
  wrapped in try/catch so the page still works without storage. It also handles hints (below).
- `games/<game>.js` gives each game the same interface:

  ```js
  export function init(puzzle)                  // → initial player state
  export function render(root, puzzle, state, marks)   // draw board; marks = rule breaks + hint highlight
  export function onInput(event, puzzle, state) // → move or null
  export function ruleBreaks(puzzle, state)     // → [{cells, reason}] visible violations, never uses solution
  export function isSolved(puzzle, state)       // → true when the board is complete and equals the solution
  export function wrongCells(puzzle, state)     // → cells whose filled value contradicts the solution
  export function stepMade(puzzle, state, step) // → true when the player's board already reflects this step
  ```

#### Controls and rule breaks per game

| Game | Controls | Live rule breaks |
|---|---|---|
| Queens | Click cycles empty → X → crown; dragging paints X | Two crowns in one row, column or region, or touching |
| Zip | Drag a path from checkpoint 1; dragging back over the path shortens it | Checkpoints out of order, or crossing a wall |
| Tango | Click cycles empty → sun → moon; givens are locked | Three in a row, more than half one symbol in a line, a broken = or × sign |
| Mini Sudoku | Click a cell, type 1–6, Backspace clears; givens are locked | A digit repeated in a row, column or box |
| Patches | Drag a rectangle; click a patch to remove it | A patch covering zero or several clues, or with the wrong area or shape |
| Wend | Drag across letters to trace a word; click a traced word to remove it | A traced path that is not a dictionary word of a remaining slot length |

Wend's dictionary check needs the word list on the client. Instead, `/api/puzzle` for Wend includes `words`:
the dictionary words that fit any path on that board (a few hundred). The board's own solver already enumerates them.

#### Hints

When you press Hint:

1. If `wrongCells` finds any cells, it highlights one and says "Something here isn't right". It doesn't say what.
2. Otherwise it shows the first step (skipping `Given` steps) where `stepMade` is false. It highlights the step's
   cell and shows `[rule] message`.
3. **Zip** has no deduction log, so its hint highlights the next cell of the answer path after the longest correct
   prefix of your path.

`stepMade` means the following:

- Queens: the cell matches the answer, so a crown where the answer has a queen, or an X where it doesn't.
- Tango and Sudoku: the cell's value equals the answer's.
- Patches: the patch covering the cell is the answer's patch.
- Wend: the answer word covering the cell has been traced.

Hints don't count against your time or best, but the win dialog shows how many hints you used.

## Error handling

- Generation failure (a generator raising `ValueError`): the pool logs it and moves on to the next seed. If
  `take` can't produce any board, it returns 503 with `{"error": "could not generate ..."}`, and the UI shows that
  message with a Retry button.
- An unknown game or level, or a bad static path, gets 400 or 404 with a JSON error, and never a stack trace.
- On the front end, a failed fetch shows an inline error and keeps the current board.

## Testing

- **Python (`tests/test_play.py`):**
  - Every `LevelSpec` produces a board whose solver finds exactly one answer (with Wend at small sizes so the test
    stays quick).
  - Pool `take` returns boards of the requested level and refills.
  - The fallback serves the closest band when a level is unreachable, using a stub generator.
  - Server: `/api/puzzle` returns the right shape, unknown game or level gets 400, and path traversal on `/static`
    gets 404.
- **JavaScript (`play/static/games/*.test.js`, run with `node --test`):** for each game, using puzzles saved as
  JSON fixtures from the Python generator:
  - the answer gives `ruleBreaks` empty, `isSolved` true and `wrongCells` empty;
  - one deliberate rule break per rule is reported;
  - `stepMade` is false for every non-Given step on the empty board and true on the solved board.
  - A pytest test runs `node --test` when `node` is on the PATH, and is skipped otherwise.
- Manual: play one board per game at each level.

## Risks

- **Wend 6×6 generation time hasn't been measured.** 5×5 takes about 4 s. If 6×6 is too slow for the pool, Hard
  stays at 5×5 and gets its difficulty from rating alone.
- **Some levels may be rare for some games** (for example, Hard Mini Sudoku). The pool fallback and the rating shown
  in the UI keep this honest instead of hanging. The level table is tuned from measured rating samples.

## Out of scope

Crossclimb and Pinpoint, a daily puzzle, reveal-answer, keyboard controls (apart from Sudoku digits), mobile and
touch layout, hosting or Docker changes, and accounts or server-side stats.
