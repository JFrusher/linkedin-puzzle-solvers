// Zip: one path through every cell, visiting the numbered checkpoints in order, never crossing a wall.
// Drag from checkpoint 1 (or from any cell already on the path) to extend; drag back over the path to shorten it.
import { adjacent, board, key, overlay, polyline, same, sameCells } from "./common.js";

export const init = () => [];

const wallSet = (puzzle) => new Set(puzzle.board.walls.map(([a, b]) => `${key(a)}|${key(b)}`));
const blocked = (walls, a, b) => walls.has(`${key(a)}|${key(b)}`) || walls.has(`${key(b)}|${key(a)}`);

export function render(root, puzzle, state, marks) {
  const { size, checkpoints, walls } = puzzle.board;
  const numbers = new Map(checkpoints.map((cell, i) => [key(cell), i + 1]));
  const onPath = new Set(state.map(key));
  const el = board(root, size, (cell, r, c) => {
    if (onPath.has(key([r, c]))) cell.classList.add("path");
    const n = numbers.get(key([r, c]));
    if (n) {
      const dot = document.createElement("span");
      dot.className = "checkpoint";
      dot.textContent = n;
      cell.append(dot);
    }
  }, marks);
  for (const [a, b] of walls) {
    const [r, c] = a[0] < b[0] || a[1] < b[1] ? a : b;
    const cell = el.children[r * size + c];
    cell.classList.add(a[0] === b[0] ? "wall-right" : "wall-bottom");
  }
  polyline(overlay(el, size), state, "zip-line");
}

export function onInput(event, puzzle, state) {
  const cell = event.cell;
  if (!cell || event.type === "up") return null;
  const first = puzzle.board.checkpoints[0];
  if (event.type === "down") {
    const at = state.findIndex((p) => same(p, cell));
    if (at >= 0) return at === state.length - 1 ? null : state.slice(0, at + 1);
    return state.length === 0 && same(cell, first) ? [cell] : null;
  }
  if (state.length === 0) return null;
  const head = state[state.length - 1];
  if (state.length > 1 && same(cell, state[state.length - 2])) return state.slice(0, -1);
  if (state.some((p) => same(p, cell)) || !adjacent(head, cell) || blocked(wallSet(puzzle), head, cell)) return null;
  return [...state, cell];
}

export function ruleBreaks(puzzle, state) {
  const order = new Map(puzzle.board.checkpoints.map((cell, i) => [key(cell), i]));
  const breaks = [];
  let expected = 0;
  for (const cell of state) {
    const n = order.get(key(cell));
    if (n === undefined) continue;
    if (n !== expected) breaks.push({ cells: [cell], reason: `checkpoint ${n + 1} reached before ${expected + 1}` });
    expected = Math.max(expected, n) + 1;
  }
  return breaks;
}

export const isSolved = (puzzle, state) => sameCells(state, puzzle.solution);

function correctPrefix(puzzle, state) {
  let n = 0;
  while (n < state.length && n < puzzle.solution.length && same(state[n], puzzle.solution[n])) n++;
  return n;
}

export function wrongCells(puzzle, state) {
  const n = correctPrefix(puzzle, state);
  return n < state.length ? [state[n]] : [];
}

/** Zip has no deduction log: the hint is the next cell of the answer path. */
export function hint(puzzle, state) {
  const n = correctPrefix(puzzle, state);
  const cell = puzzle.solution[n];
  return cell ? { cell, text: n === 0 ? "Start on checkpoint 1." : `Continue the path to (${cell[0]}, ${cell[1]}).` } : null;
}

export const stepMade = () => true;
