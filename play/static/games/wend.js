// Wend: split every letter tile into words of the slot lengths, each traced through side-by-side tiles.
// Drag across letters to trace a word; click a traced word to remove it.
import { adjacent, board, key, overlay, polyline, same, sameCells, sameSet } from "./common.js";

const COLOURS = ["#a2d2ff", "#bde0a8", "#ffd6a5", "#cdb4db", "#f9c6e0", "#a0e7e5", "#ffb4a2", "#fdffb6"];
let tracing = null; // the word being dragged; UI only

export function init() {
  tracing = null;
  return [];
}

const spell = (puzzle, path) => path.map(([r, c]) => puzzle.board.letters[r][c]).join("");
const isWord = (puzzle, path) =>
  puzzle.placements.some((p) => sameCells(p, path) || sameCells(p, [...path].reverse()));

export function render(root, puzzle, state, marks) {
  const { size, letters, word_lengths: lengths } = puzzle.board;
  const owner = new Map(state.flatMap((path, i) => path.map((cell) => [key(cell), i])));
  const el = board(root, size, (cell, r, c) => {
    if (letters[r][c] === null) {
      cell.classList.add("wall");
      // Outline only the bar's outer edges, as LinkedIn draws a run of walls as one block.
      const open = (dr, dc) => letters[r + dr]?.[c + dc] !== undefined && letters[r + dr][c + dc] !== null;
      const edges = [[-1, 0, "0 3px"], [1, 0, "0 -3px"], [0, -1, "3px 0"], [0, 1, "-3px 0"]];
      cell.style.boxShadow = edges.filter(([dr, dc]) => open(dr, dc)).map(([, , d]) => `inset ${d} 0 var(--frame)`).join(", ");
      return;
    }
    cell.textContent = letters[r][c];
    const i = owner.get(key([r, c]));
    if (i !== undefined) Object.assign(cell.style, { background: COLOURS[i % COLOURS.length], color: "#1d1d1d" });
    if (tracing?.some((p) => same(p, [r, c]))) cell.classList.add("tracing");
  }, marks);
  const svg = overlay(el, size);
  for (const path of state) polyline(svg, path, "word-line");
  if (tracing) polyline(svg, tracing, "word-line live");

  const slots = document.createElement("div");
  slots.className = "slots";
  const found = state.filter((path) => isWord(puzzle, path)).map((path) => spell(puzzle, path));
  for (const length of [...lengths].sort((a, b) => a - b)) {
    const at = found.findIndex((word) => word.length === length);
    const word = at >= 0 ? found.splice(at, 1)[0] : "";
    const row = document.createElement("div");
    row.className = "slot";
    for (let i = 0; i < length; i++) {
      const box = document.createElement("span");
      box.textContent = word[i] ?? "";
      row.append(box);
    }
    slots.append(row);
  }
  root.append(slots);
}

export function onInput(event, puzzle, state) {
  const cell = event.cell;
  const open = cell && puzzle.board.letters[cell[0]][cell[1]] !== null;
  if (event.type === "down") {
    tracing = open && !state.some((path) => path.some((p) => same(p, cell))) ? [cell] : null;
    if (!tracing && open) {
      const hit = state.find((path) => path.some((p) => same(p, cell)));
      return state.filter((path) => path !== hit);
    }
    return state;
  }
  if (!tracing) return null;
  if (event.type === "move" && open) {
    const head = tracing[tracing.length - 1];
    if (tracing.length > 1 && same(cell, tracing[tracing.length - 2])) tracing = tracing.slice(0, -1);
    else if (adjacent(head, cell) && !tracing.some((p) => same(p, cell)) && !state.some((path) => path.some((p) => same(p, cell))))
      tracing = [...tracing, cell];
    return state;
  }
  if (event.type !== "up") return null;
  const made = tracing;
  tracing = null;
  return made.length > 1 ? [...state, made] : state;
}

export function ruleBreaks(puzzle, state) {
  return state
    .filter((path) => !isWord(puzzle, path))
    .map((path) => ({ cells: path, reason: `"${spell(puzzle, path)}" is not a word that fits a slot` }));
}

const answerFor = (puzzle, path) => puzzle.solution.find((w) => sameSet(w.cells, path));

export const isSolved = (puzzle, state) =>
  state.length === puzzle.solution.length && state.every((path) => answerFor(puzzle, path));

export const wrongCells = (puzzle, state) => state.filter((path) => !answerFor(puzzle, path)).map((path) => path[0]);

export function stepMade(puzzle, state, step) {
  const want = puzzle.solution.find((w) => w.cells.some((p) => same(p, step.cell)));
  return state.some((path) => sameSet(path, want.cells));
}
