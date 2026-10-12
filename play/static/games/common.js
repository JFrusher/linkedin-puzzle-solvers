// Shared helpers for the game modules: cell keys, board DOM and SVG overlays. No DOM access at import time,
// so the rule functions can be tested under `node --test`.

export const key = ([r, c]) => `${r},${c}`;
export const same = (a, b) => a[0] === b[0] && a[1] === b[1];
export const adjacent = (a, b) => Math.abs(a[0] - b[0]) + Math.abs(a[1] - b[1]) === 1;
export const grid = (size, fill) => Array.from({ length: size }, (_, r) => Array.from({ length: size }, (_, c) => fill(r, c)));
export const sameCells = (a, b) => a.length === b.length && a.every((cell, i) => same(cell, b[i]));
export function sameSet(a, b) {
  const keys = new Set(a.map(key));
  return a.length === b.length && keys.size === a.length && b.every((cell) => keys.has(key(cell)));
}

/** Lines of an n×n board: rows then columns, each a list of cells. */
export function lines(size) {
  const rows = grid(size, (r, c) => [r, c]);
  return [...rows, ...rows[0].map((_, c) => rows.map((row) => row[c]))];
}

/** Board element: a CSS grid of cells carrying data-r / data-c, filled by `paint(cellEl, r, c)`. */
export function board(root, size, paint, marks) {
  const el = document.createElement("div");
  el.className = "board";
  el.style.setProperty("--n", size);
  for (let r = 0; r < size; r++) {
    for (let c = 0; c < size; c++) {
      const cell = document.createElement("div");
      cell.className = "cell";
      cell.dataset.r = r;
      cell.dataset.c = c;
      paint(cell, r, c);
      if (marks.bad.has(key([r, c]))) cell.classList.add("bad");
      if (marks.hint === key([r, c])) cell.classList.add("hint");
      el.append(cell);
    }
  }
  root.replaceChildren(el);
  return el;
}

/** An SVG laid over the board in cell units (cell centres at r + 0.5, c + 0.5). */
export function overlay(boardEl, size) {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", `0 0 ${size} ${size}`);
  svg.classList.add("overlay");
  boardEl.append(svg);
  return svg;
}

export function polyline(svg, cells, className) {
  if (cells.length === 0) return;
  const line = document.createElementNS("http://www.w3.org/2000/svg", "polyline");
  line.setAttribute("points", cells.map(([r, c]) => `${c + 0.5},${r + 0.5}`).join(" "));
  line.setAttribute("class", className);
  svg.append(line);
}

/** An absolutely placed element over the board, in cell units. */
export function place(boardEl, size, { top, left, height = 1, width = 1 }, className, text = "") {
  const el = document.createElement("div");
  el.className = className;
  Object.assign(el.style, {
    top: `${(top / size) * 100}%`,
    left: `${(left / size) * 100}%`,
    height: `${(height / size) * 100}%`,
    width: `${(width / size) * 100}%`,
  });
  el.textContent = text;
  boardEl.append(el);
  return el;
}
