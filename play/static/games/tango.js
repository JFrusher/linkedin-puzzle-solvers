// Tango: fill every cell with a sun or a moon. No three alike in a row, each line half suns and half moons,
// "=" joins equal cells and "×" opposite ones. Click cycles empty → sun → moon; given cells are locked.
import { board, grid, lines, place } from "./common.js";

const ICON = { sun: "☀", moon: "☾" };
/** A moon is two clicks away (empty → sun → moon), so new rule breaks wait this long before showing. */
export const GRACE_MS = 1000;
const NEXT = { null: "sun", sun: "moon", moon: null };

export const init = (puzzle) => grid(puzzle.board.size, (r, c) => puzzle.board.givens[r][c]);

export function render(root, puzzle, state, marks) {
  const { size, givens, constraints } = puzzle.board;
  const el = board(root, size, (cell, r, c) => {
    const value = state[r][c];
    if (value) {
      const icon = document.createElement("span");
      icon.className = `symbol ${value}`;
      icon.textContent = ICON[value];
      cell.append(icon);
    }
    if (givens[r][c]) cell.classList.add("given");
  }, marks);
  for (const { a, b, relation } of constraints) {
    const top = (a[0] + b[0]) / 2 + 0.5 - 0.15;
    const left = (a[1] + b[1]) / 2 + 0.5 - 0.15;
    place(el, size, { top, left, height: 0.3, width: 0.3 }, "sign", relation === "=" ? "=" : "×");
  }
}

export function onInput(event, puzzle, state) {
  if (event.type !== "up" || !event.cell) return null;
  const [r, c] = event.cell;
  if (puzzle.board.givens[r][c]) return null;
  const next = state.map((row) => [...row]);
  next[r][c] = NEXT[state[r][c]];
  return next;
}

export function ruleBreaks(puzzle, state) {
  const { size, constraints } = puzzle.board;
  const breaks = [];
  for (const line of lines(size)) {
    for (let i = 0; i + 2 < line.length; i++) {
      const run = line.slice(i, i + 3);
      const [first] = run.map(([r, c]) => state[r][c]);
      if (first && run.every(([r, c]) => state[r][c] === first)) breaks.push({ cells: run, reason: `three ${first}s in a row` });
    }
    for (const symbol of ["sun", "moon"]) {
      const here = line.filter(([r, c]) => state[r][c] === symbol);
      if (here.length > size / 2) breaks.push({ cells: here, reason: `more than ${size / 2} ${symbol}s in one line` });
    }
  }
  for (const { a, b, relation } of constraints) {
    const [x, y] = [state[a[0]][a[1]], state[b[0]][b[1]]];
    if (x && y && (relation === "=") !== (x === y)) breaks.push({ cells: [a, b], reason: `broken "${relation}" sign` });
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
