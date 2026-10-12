// Patches: cover the grid with rectangles, each holding exactly one clue and matching its area and shape.
// Drag a rectangle (a click makes a 1×1); a new patch replaces any it overlaps; click a patch to remove it.
import { board, place } from "./common.js";

const COLOURS = ["#ffb4a2", "#a2d2ff", "#bde0a8", "#ffd6a5", "#cdb4db", "#f9c6e0", "#a0e7e5", "#fdffb6", "#c3c3f5"];
const SHAPE_ICON = { square: "■", tall: "▮", wide: "▬", any: "" };

export const init = () => [];

const rect = (a, b) => ({
  top: Math.min(a[0], b[0]),
  left: Math.min(a[1], b[1]),
  height: Math.abs(a[0] - b[0]) + 1,
  width: Math.abs(a[1] - b[1]) + 1,
});
const covers = (p, [r, c]) => r >= p.top && r < p.top + p.height && c >= p.left && c < p.left + p.width;
const overlaps = (p, q) =>
  p.top < q.top + q.height && q.top < p.top + p.height && p.left < q.left + q.width && q.left < p.left + p.width;
const equal = (p, q) => p.top === q.top && p.left === q.left && p.height === q.height && p.width === q.width;
const shapeOf = (p) => (p.height === p.width ? "square" : p.height > p.width ? "tall" : "wide");

let preview = null; // the rectangle being dragged; UI only

export function render(root, puzzle, state, marks) {
  const { size, clues } = puzzle.board;
  const el = board(root, size, () => {}, marks);
  state.forEach((p, i) => place(el, size, p, "patch").style.setProperty("--patch", COLOURS[i % COLOURS.length]));
  if (preview) place(el, size, preview, "patch preview");
  for (const clue of clues) {
    const badge = place(el, size, { top: clue.cell[0] + 0.2, left: clue.cell[1] + 0.2, height: 0.6, width: 0.6 }, `badge ${clue.shape}`);
    badge.textContent = clue.area ?? SHAPE_ICON[clue.shape];
  }
}

export function onInput(event, puzzle, state, drag) {
  if (event.type === "down" && event.cell) {
    drag.start = event.cell;
    preview = rect(event.cell, event.cell);
    return state;
  }
  if (!drag.start) return null;
  if (event.type === "move" && event.cell) {
    preview = rect(drag.start, event.cell);
    return state;
  }
  if (event.type !== "up") return null;
  const made = preview;
  preview = null;
  if (made.height === 1 && made.width === 1) {
    const hit = state.find((p) => covers(p, drag.start));
    if (hit) return state.filter((p) => p !== hit);
  }
  return [...state.filter((p) => !overlaps(p, made)), made];
}

export function ruleBreaks(puzzle, state) {
  const breaks = [];
  for (const p of state) {
    const inside = puzzle.board.clues.filter((clue) => covers(p, clue.cell));
    const cells = [];
    for (let r = p.top; r < p.top + p.height; r++) for (let c = p.left; c < p.left + p.width; c++) cells.push([r, c]);
    if (inside.length !== 1) {
      breaks.push({ cells, reason: inside.length ? "a patch holds more than one clue" : "a patch holds no clue" });
      continue;
    }
    const [clue] = inside;
    if (clue.area !== null && clue.area !== p.height * p.width) breaks.push({ cells, reason: `this patch should cover ${clue.area} cells` });
    if (clue.shape !== "any" && clue.shape !== shapeOf(p)) breaks.push({ cells, reason: `this patch should be ${clue.shape}` });
  }
  return breaks;
}

const inAnswer = (puzzle, p) => puzzle.solution.some((q) => equal(p, q));

export const isSolved = (puzzle, state) => state.length === puzzle.solution.length && state.every((p) => inAnswer(puzzle, p));

export function wrongCells(puzzle, state) {
  return state.filter((p) => !inAnswer(puzzle, p)).map((p) => [p.top, p.left]);
}

export function stepMade(puzzle, state, step) {
  const want = puzzle.solution.find((q) => covers(q, step.cell));
  return state.some((p) => equal(p, want));
}
