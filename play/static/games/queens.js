// Queens: one crown per row, column and colour region; no two crowns touch, even diagonally.
// Click cycles empty → X → crown; dragging paints X over empty cells.
import { board, grid, key, lines } from "./common.js";

const EMPTY = 0, X = 1, CROWN = 2;
const COLOURS = ["#bba3e2", "#ffc992", "#96beff", "#b3dfa0", "#dfdfdf", "#ff7b60", "#e6f388", "#b9b29e", "#dfa0bf", "#a3d2d8", "#62efea"];

export const init = (puzzle) => grid(puzzle.board.size, () => EMPTY);

export function render(root, puzzle, state, marks) {
  const { regions } = puzzle.board;
  board(root, puzzle.board.size, (cell, r, c) => {
    cell.style.background = COLOURS[regions[r][c] % COLOURS.length];
    cell.style.color = "#1d1d1d"; // light region colours in both themes
    const edge = (dr, dc) => regions[r + dr]?.[c + dc] !== undefined && regions[r + dr][c + dc] !== regions[r][c];
    if (edge(0, 1)) cell.classList.add("edge-right");
    if (edge(1, 0)) cell.classList.add("edge-bottom");
    cell.textContent = state[r][c] === CROWN ? "♛" : state[r][c] === X ? "×" : "";
    cell.classList.toggle("crown", state[r][c] === CROWN);
  }, marks);
}

export function onInput(event, puzzle, state, drag) {
  if (!event.cell) return null;
  const [r, c] = event.cell;
  if (event.type === "down") {
    drag.start = event.cell;
    return null;
  }
  if (event.type === "move" && drag.start && key(event.cell) !== key(drag.start)) {
    drag.moved = true;
    const next = state.map((row) => [...row]);
    for (const [pr, pc] of [drag.start, event.cell]) if (next[pr][pc] === EMPTY) next[pr][pc] = X;
    return next;
  }
  if (event.type === "up" && !drag.moved && drag.start && key(event.cell) === key(drag.start)) {
    const next = state.map((row) => [...row]);
    next[r][c] = (next[r][c] + 1) % 3;
    return next;
  }
  return null;
}

const crowns = (state) => state.flatMap((row, r) => row.flatMap((v, c) => (v === CROWN ? [[r, c]] : [])));

export function ruleBreaks(puzzle, state) {
  const breaks = [];
  const placed = crowns(state);
  const groups = [...lines(puzzle.board.size).map((line, i) => [i < puzzle.board.size ? "row" : "column", line])];
  const regions = new Map();
  puzzle.board.regions.forEach((row, r) => row.forEach((id, c) => regions.set(id, [...(regions.get(id) ?? []), [r, c]])));
  for (const cells of regions.values()) groups.push(["colour region", cells]);
  for (const [name, cells] of groups) {
    const here = cells.filter(([r, c]) => state[r][c] === CROWN);
    if (here.length > 1) breaks.push({ cells: here, reason: `two crowns in one ${name}` });
  }
  for (let i = 0; i < placed.length; i++) {
    for (let j = i + 1; j < placed.length; j++) {
      const [a, b] = [placed[i], placed[j]];
      if (Math.abs(a[0] - b[0]) <= 1 && Math.abs(a[1] - b[1]) <= 1) breaks.push({ cells: [a, b], reason: "crowns touch" });
    }
  }
  return breaks;
}

const answer = (puzzle) => new Set(puzzle.solution.map(key));

export function isSolved(puzzle, state) {
  const placed = crowns(state);
  const queens = answer(puzzle);
  return placed.length === queens.size && placed.every((cell) => queens.has(key(cell)));
}

export function wrongCells(puzzle, state) {
  const queens = answer(puzzle);
  return state.flatMap((row, r) =>
    row.flatMap((v, c) => ((v === CROWN && !queens.has(key([r, c]))) || (v === X && queens.has(key([r, c]))) ? [[r, c]] : [])),
  );
}

export function stepMade(puzzle, state, step) {
  const [r, c] = step.cell;
  return answer(puzzle).has(key(step.cell)) ? state[r][c] === CROWN : state[r][c] === X;
}
