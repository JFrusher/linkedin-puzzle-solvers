// Rule checks for the play page's game modules, against puzzles saved from the Python generators.
// Run: node --test tests/play_js   (tests/test_play.py runs this too when node is installed)
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

const games = ["queens", "zip", "tango", "mini-sudoku", "patches", "wend"];
const fixture = (game) => JSON.parse(readFileSync(new URL(`../fixtures/play/${game}.json`, import.meta.url)));
const load = (game) => import(`../../play/static/games/${game}.js`);

function solvedState(game, puzzle) {
  const { solution, board } = puzzle;
  if (game === "queens") {
    const crowns = new Set(solution.map(([r, c]) => `${r},${c}`));
    return Array.from({ length: board.size }, (_, r) => Array.from({ length: board.size }, (_, c) => (crowns.has(`${r},${c}`) ? 2 : 1)));
  }
  if (game === "patches") return solution.map(({ top, left, height, width }) => ({ top, left, height, width }));
  if (game === "wend") return solution.map((word) => word.cells);
  return structuredClone(solution);
}

for (const game of games) {
  test(`${game}: the answer breaks no rule, is solved and has no wrong cells`, async () => {
    const module = await load(game);
    const puzzle = fixture(game);
    const state = solvedState(game, puzzle);
    assert.deepEqual(module.ruleBreaks(puzzle, state), []);
    assert.equal(module.isSolved(puzzle, state), true);
    assert.deepEqual(module.wrongCells(puzzle, state), []);
  });

  test(`${game}: a fresh board is unsolved, error-free, and has every step still to make`, async () => {
    const module = await load(game);
    const puzzle = fixture(game);
    const state = module.init(puzzle);
    assert.equal(module.isSolved(puzzle, state), false);
    assert.deepEqual(module.ruleBreaks(puzzle, state), []);
    assert.deepEqual(module.wrongCells(puzzle, state), []);
    if (game === "zip") return; // Zip hints follow the path, not the log
    const steps = puzzle.steps.filter((step) => step.rule !== "Given" && step.cell);
    assert.ok(steps.length > 0);
    for (const step of steps) assert.equal(module.stepMade(puzzle, state, step), false, step.message);
    const solved = solvedState(game, puzzle);
    for (const step of steps) assert.equal(module.stepMade(puzzle, solved, step), true, step.message);
  });
}

const reasons = (breaks) => breaks.map((b) => b.reason);

test("queens: crowns sharing a row or touching are rule breaks; click cycles empty → X → crown", async () => {
  const queens = await load("queens");
  const puzzle = { board: { size: 4, regions: [[0, 0, 1, 1], [0, 0, 1, 1], [2, 2, 3, 3], [2, 2, 3, 3]] }, solution: [] };
  const state = queens.init(puzzle);
  state[0][0] = 2;
  state[0][3] = 2;
  assert.deepEqual(reasons(queens.ruleBreaks(puzzle, state)), ["two crowns in one row"]);
  state[0][3] = 0;
  state[1][1] = 2;
  assert.deepEqual(reasons(queens.ruleBreaks(puzzle, state)), ["two crowns in one colour region", "crowns touch"]);

  let cell = queens.init(puzzle);
  for (const expected of [1, 2, 0]) {
    const drag = {};
    queens.onInput({ type: "down", cell: [2, 2] }, puzzle, cell, drag);
    cell = queens.onInput({ type: "up", cell: [2, 2] }, puzzle, cell, drag);
    assert.equal(cell[2][2], expected);
  }
});

test("zip: reaching a later checkpoint first is a rule break; walls block and dragging back shortens", async () => {
  const zip = await load("zip");
  const puzzle = { board: { size: 3, checkpoints: [[0, 0], [0, 2], [2, 2]], walls: [[[0, 0], [0, 1]]] }, solution: [] };
  const early = [[0, 0], [1, 0], [2, 0], [2, 1], [2, 2]];
  assert.deepEqual(reasons(zip.ruleBreaks(puzzle, early)), ["checkpoint 3 reached before 2"]);

  let path = zip.onInput({ type: "down", cell: [0, 0] }, puzzle, zip.init());
  assert.equal(zip.onInput({ type: "move", cell: [0, 1] }, puzzle, path), null, "wall between (0,0) and (0,1)");
  path = zip.onInput({ type: "move", cell: [1, 0] }, puzzle, path);
  path = zip.onInput({ type: "move", cell: [1, 1] }, puzzle, path);
  assert.deepEqual(zip.onInput({ type: "move", cell: [1, 0] }, puzzle, path), [[0, 0], [1, 0]]);
});

test("tango: three in a row, too many of one symbol and broken signs are rule breaks", async () => {
  const tango = await load("tango");
  const empty = Array.from({ length: 6 }, () => Array(6).fill(null));
  const puzzle = { board: { size: 6, givens: empty, constraints: [{ a: [5, 0], b: [5, 1], relation: "=" }] } };
  const state = structuredClone(empty);
  state[0] = ["sun", "sun", "sun", "moon", "sun", null];
  state[5][0] = "sun";
  state[5][1] = "moon";
  assert.deepEqual(reasons(tango.ruleBreaks(puzzle, state)), ["three suns in a row", "more than 3 suns in one line", 'broken "=" sign']);
});

test("mini sudoku: a repeated digit is a rule break; typing fills the selected cell", async () => {
  const sudoku = await load("mini-sudoku");
  const givens = Array.from({ length: 6 }, () => Array(6).fill(0));
  const puzzle = { board: { size: 6, givens } };
  let state = sudoku.init(puzzle);
  state = sudoku.onInput({ type: "up", cell: [0, 0] }, puzzle, state);
  state = sudoku.onInput({ type: "key", key: "4" }, puzzle, state);
  state = sudoku.onInput({ type: "up", cell: [0, 5] }, puzzle, state);
  state = sudoku.onInput({ type: "key", key: "4" }, puzzle, state);
  assert.deepEqual(reasons(sudoku.ruleBreaks(puzzle, state)), ["4 twice in one row"]);
});

test("patches: patches with no clue, two clues or the wrong area or shape are rule breaks", async () => {
  const patches = await load("patches");
  const clues = [{ cell: [0, 0], area: 2, shape: "any" }, { cell: [0, 1], area: null, shape: "tall" }, { cell: [2, 2], area: null, shape: "any" }];
  const puzzle = { board: { size: 3, clues } };
  const state = [{ top: 0, left: 0, height: 1, width: 1 }, { top: 0, left: 1, height: 1, width: 2 }, { top: 1, left: 0, height: 1, width: 1 }];
  assert.deepEqual(reasons(patches.ruleBreaks(puzzle, state)), [
    "this patch should cover 2 cells",
    "this patch should be tall",
    "a patch holds no clue",
  ]);
});

test("wend: a traced path that is not a word is a rule break, in either direction a word is fine", async () => {
  const wend = await load("wend");
  const puzzle = fixture("wend");
  const word = puzzle.solution[0].cells;
  assert.deepEqual(wend.ruleBreaks(puzzle, [word, [...word].reverse()]), []);
  const scrambled = [word[1], word[0], ...word.slice(2)];
  const breaks = wend.ruleBreaks(puzzle, [scrambled]);
  assert.equal(breaks.length, 1);
});
