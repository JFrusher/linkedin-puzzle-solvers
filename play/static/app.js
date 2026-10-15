// Shell shared by every game: fetching puzzles, pointer and key input, undo / redo, timer, bests and hints.
// Each game module (games/<name>.js) supplies init, render, onInput, ruleBreaks, isSolved, wrongCells, stepMade.
import { key } from "./games/common.js";

const GAMES = {
  queens: ["Queens", "Click: empty → X → crown. Drag to fill X's. One crown per row, column and colour; crowns never touch."],
  zip: ["Zip", "Drag from 1 through every cell, visiting the numbers in order. Drag back to undo a stretch; walls block."],
  tango: ["Tango", "Click: sun → moon → empty. No three alike in a row; each line half suns; = means same, × means opposite."],
  "mini-sudoku": ["Mini Sudoku", "Click a cell, type 1–6 (Backspace clears). Each row, column and 2×3 box holds 1–6 once."],
  patches: ["Patches", "Drag rectangles. Each holds one clue: its number is the area, its shape square / tall / wide."],
  wend: ["Wend", "Drag across letters to trace words of the slot lengths, using every tile once. Click a word to remove it."],
};

const $ = (id) => document.getElementById(id);
const boardRoot = $("board-root");
const play = { game: "queens", level: "easy", loadedGame: "queens", loadedLevel: "easy", module: null, puzzle: null, state: null, history: [], future: [],
  version: 0, started: null, finished: null, hints: 0, hint: null, lastHint: null, loading: 0,
  shown: new Set(), graceUntil: 0 };

const store = {
  get(name, fallback) { try { return localStorage.getItem(name) ?? fallback; } catch { return fallback; } },
  set(name, value) { try { localStorage.setItem(name, value); } catch { /* storage blocked: bests just aren't kept */ } },
};

function seconds() {
  if (!play.started) return 0;
  return Math.floor(((play.finished ?? Date.now()) - play.started) / 1000);
}
const clock = (s) => `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;

const breakId = (b) => `${b.reason}|${b.cells.map(key).sort().join(" ")}`;

function draw() {
  if (!play.puzzle) return;
  let breaks = play.module.ruleBreaks(play.puzzle, play.state);
  // Grace: right after a move, only breaks already on screen stay; new ones wait, so passing through a state
  // (Tango's sun on the way to a moon) doesn't flash red.
  if (Date.now() < play.graceUntil) breaks = breaks.filter((b) => play.shown.has(breakId(b)));
  play.shown = new Set(breaks.map(breakId));
  const marks = { bad: new Set(breaks.flatMap((b) => b.cells.map(key))), hint: play.hint ? key(play.hint.cell) : null };
  play.module.render(boardRoot, play.puzzle, play.state, marks);
  $("problems").replaceChildren(...[...new Set(breaks.map((b) => b.reason))].map((reason) => {
    const li = document.createElement("li");
    li.textContent = reason;
    return li;
  }));
  $("hint").hidden = !play.hint;
  $("hint").textContent = play.hint?.text ?? "";
  $("undo").disabled = play.history.length === 0;
  $("redo").disabled = play.future.length === 0;
}

async function load() {
  const ticket = ++play.loading;
  const [title, help] = GAMES[play.game];
  document.querySelectorAll("#games button").forEach((b) => b.classList.toggle("active", b.dataset.game === play.game));
  document.querySelectorAll(".levels button").forEach((b) => b.classList.toggle("active", b.dataset.level === play.level));
  $("help").textContent = help;
  $("rating").textContent = `Generating ${title}…`;
  boardRoot.replaceChildren();
  try {
    const response = await fetch(`/api/puzzle?game=${play.game}&level=${play.level}`);
    const body = await response.json();
    if (!response.ok) throw new Error(body.error ?? response.statusText);
    const module = await import(`./games/${play.game}.js`);
    if (ticket !== play.loading) return; // a newer request superseded this one
    Object.assign(play, { loadedGame: play.game, loadedLevel: play.level, module, puzzle: body, state: module.init(body), history: [], future: [], version: 0,
      started: null, finished: null, hints: 0, hint: null, lastHint: null, shown: new Set(), graceUntil: 0 });
    $("rating").textContent = body.rating;
    draw();
  } catch (error) {
    if (ticket !== play.loading) return;
    $("rating").textContent = "";
    draw();
    const retry = document.createElement("button");
    retry.textContent = "Retry";
    retry.onclick = load;
    const message = document.createElement("p");
    message.className = "error";
    message.textContent = `Could not load a puzzle: ${error.message} `;
    message.append(retry);
    boardRoot.append(message);
  }
}

// --- moves -------------------------------------------------------------------------------------------------

let gesture = null; // {before, drag} while a pointer is down

function apply(next) {
  if (next === null || next === undefined) return;
  play.state = next;
  draw();
}

function commit(before) {
  if (JSON.stringify(before) === JSON.stringify(play.state)) return draw();
  play.history.push(before);
  play.future = [];
  play.version++;
  play.hint = null;
  play.started ??= Date.now();
  if (play.module.GRACE_MS) {
    play.graceUntil = Date.now() + play.module.GRACE_MS;
    setTimeout(draw, play.module.GRACE_MS + 10);
  }
  if (!play.finished && play.module.isSolved(play.puzzle, play.state)) win();
  draw();
}

function cellAt(event) {
  const el = document.elementFromPoint(event.clientX, event.clientY)?.closest(".cell");
  return el && boardRoot.contains(el) ? [Number(el.dataset.r), Number(el.dataset.c)] : null;
}

boardRoot.addEventListener("pointerdown", (event) => {
  if (!play.puzzle || play.finished) return;
  const cell = cellAt(event);
  if (!cell) return;
  event.preventDefault();
  boardRoot.setPointerCapture(event.pointerId);
  gesture = { before: play.state, drag: {}, last: key(cell) };
  apply(play.module.onInput({ type: "down", cell }, play.puzzle, play.state, gesture.drag));
});

boardRoot.addEventListener("pointermove", (event) => {
  if (!gesture) return;
  const cell = cellAt(event);
  if (!cell || key(cell) === gesture.last) return;
  gesture.last = key(cell);
  apply(play.module.onInput({ type: "move", cell }, play.puzzle, play.state, gesture.drag));
});

function endGesture(event) {
  if (!gesture) return;
  const { before, drag } = gesture;
  gesture = null;
  apply(play.module.onInput({ type: "up", cell: cellAt(event) }, play.puzzle, play.state, drag));
  commit(before);
}
boardRoot.addEventListener("pointerup", endGesture);
boardRoot.addEventListener("pointercancel", endGesture);

document.addEventListener("keydown", (event) => {
  if (event.ctrlKey && event.key.toLowerCase() === "z") return void (event.preventDefault(), undo());
  if (event.ctrlKey && event.key.toLowerCase() === "y") return void (event.preventDefault(), redo());
  if (!play.puzzle || play.finished || event.ctrlKey || event.metaKey || event.altKey) return;
  const before = play.state;
  const next = play.module.onInput({ type: "key", key: event.key }, play.puzzle, play.state, {});
  if (next) {
    event.preventDefault();
    apply(next);
    commit(before);
  }
});

function undo() {
  if (!play.history.length || play.finished) return;
  play.future.push(play.state);
  play.state = play.history.pop();
  play.version++;
  play.hint = null;
  draw();
}

function redo() {
  if (!play.future.length || play.finished) return;
  play.history.push(play.state);
  play.state = play.future.pop();
  play.version++;
  play.hint = null;
  if (play.module.isSolved(play.puzzle, play.state)) win();
  draw();
}

function clear() {
  if (!play.puzzle || play.finished) return;
  const before = play.state;
  play.state = play.module.init(play.puzzle);
  commit(before);
}

// --- hints and winning -------------------------------------------------------------------------------------

function hint() {
  if (!play.puzzle || play.finished) return;
  const { module, puzzle, state } = play;
  const wrong = module.wrongCells(puzzle, state);
  if (wrong.length) {
    play.hint = { cell: wrong[0], text: "Something here isn't right." };
  } else if (module.hint) {
    play.hint = module.hint(puzzle, state);
  } else {
    const open = puzzle.steps.filter((step) => step.rule !== "Given" && step.cell && !module.stepMade(puzzle, state, step));
    // Asking again without moving shows the next unmade step, not the same one.
    const again = play.lastHint?.version === play.version ? open.findIndex((s) => s.index > play.lastHint.index) : 0;
    const step = open[again >= 0 ? again : 0];
    play.hint = step ? { cell: step.cell, text: `${step.rule}: ${step.message}` } : null;
    play.lastHint = step ? { version: play.version, index: step.index } : null;
  }
  if (play.hint) play.hints++;
  draw();
}

function win() {
  play.finished = Date.now();
  play.started ??= play.finished;
  const time = seconds();
  const name = `play:best:${play.loadedGame}:${play.loadedLevel}`;
  const best = Number(store.get(name, "0"));
  const record = !best || time < best;
  if (record) store.set(name, String(time));
  const hints = play.hints ? ` with ${play.hints} hint${play.hints > 1 ? "s" : ""}` : "";
  $("win-text").textContent =
    `${GAMES[play.loadedGame][0]} (${play.loadedLevel}) in ${clock(time)}${hints}. ` +
    (record ? (best ? `New best (was ${clock(best)})!` : "First solve at this level!") : `Best: ${clock(best)}.`);
  $("win").showModal();
}

setInterval(() => { $("timer").textContent = clock(seconds()); }, 250);

// --- controls ----------------------------------------------------------------------------------------------

for (const [name, [title]] of Object.entries(GAMES)) {
  const button = document.createElement("button");
  button.textContent = title;
  button.dataset.game = name;
  button.onclick = () => { play.game = name; store.set("play:game", name); load(); };
  $("games").append(button);
}
document.querySelectorAll(".levels button").forEach((button) => {
  button.onclick = () => { play.level = button.dataset.level; store.set("play:level", play.level); load(); };
});
$("undo").onclick = undo;
$("redo").onclick = redo;
$("clear").onclick = clear;
$("hint-button").onclick = hint;
$("new").onclick = load;
$("win-new").onclick = load;

play.game = GAMES[store.get("play:game", "")] ? store.get("play:game", "") : "queens";
play.level = ["easy", "medium", "hard"].includes(store.get("play:level", "")) ? store.get("play:level", "") : "easy";
load();
