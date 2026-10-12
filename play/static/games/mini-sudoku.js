// Mini Sudoku: 1–6 once in every row, column and 2×3 box. Click a cell, then type a digit; Backspace clears.
import { board, grid, lines } from "./common.js";

const SIZE = 6, BOX_ROWS = 2, BOX_COLS = 3;
let selected = null; // UI only, so not part of the undoable state

export function init(puzzle) {
  selected = null;
  return grid(SIZE, (r, c) => puzzle.board.givens[r][c]);
}

export function render(root, puzzle, state, marks) {
  board(root, SIZE, (cell, r, c) => {
    cell.textContent = state[r][c] || "";
    if (puzzle.board.givens[r][c]) cell.classList.add("given");
    if (c % BOX_COLS === BOX_COLS - 1 && c < SIZE - 1) cell.classList.add("edge-right");
    if (r % BOX_ROWS === BOX_ROWS - 1 && r < SIZE - 1) cell.classList.add("edge-bottom");
    if (selected && selected[0] === r && selected[1] === c) cell.classList.add("selected");
  }, marks);
}

/** Selecting a cell changes no state, so it returns the same state to trigger a redraw without an undo step. */
export function onInput(event, puzzle, state) {
  if (event.type === "up" && event.cell) {
    selected = puzzle.board.givens[event.cell[0]][event.cell[1]] ? null : event.cell;
    return state;
  }
  if (event.type !== "key" || !selected) return null;
  const digit = event.key >= "1" && event.key <= "6" ? Number(event.key) : ["Backspace", "Delete", "0"].includes(event.key) ? 0 : null;
  if (digit === null) return null;
  const next = state.map((row) => [...row]);
  next[selected[0]][selected[1]] = digit;
  return next;
}

function units() {
  const boxes = [];
  for (let br = 0; br < SIZE; br += BOX_ROWS) {
    for (let bc = 0; bc < SIZE; bc += BOX_COLS) {
      const box = [];
      for (let r = br; r < br + BOX_ROWS; r++) for (let c = bc; c < bc + BOX_COLS; c++) box.push([r, c]);
      boxes.push(["box", box]);
    }
  }
  return [...lines(SIZE).map((line, i) => [i < SIZE ? "row" : "column", line]), ...boxes];
}

export function ruleBreaks(puzzle, state) {
  const breaks = [];
  for (const [name, cells] of units()) {
    for (let d = 1; d <= SIZE; d++) {
      const here = cells.filter(([r, c]) => state[r][c] === d);
      if (here.length > 1) breaks.push({ cells: here, reason: `${d} twice in one ${name}` });
    }
  }
  return breaks;
}

export const isSolved = (puzzle, state) => state.every((row, r) => row.every((v, c) => v === puzzle.solution[r][c]));

export const wrongCells = (puzzle, state) =>
  state.flatMap((row, r) => row.flatMap((v, c) => (v && v !== puzzle.solution[r][c] ? [[r, c]] : [])));

export function stepMade(puzzle, state, step) {
  const [r, c] = step.cell;
  return state[r][c] === puzzle.solution[r][c];
}
