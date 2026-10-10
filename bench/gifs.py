"""Render the README's "How it works" GIFs from the real screenshots.

    python -m bench.gifs            # writes docs/gifs/<game>.gif

Each GIF walks one real board through the pipeline: find the board, detect the grid, read the cells, then
replay the solver's own deduction log step by step and finish on the solved screenshot.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from bench import real
from core import visualizer
from core.vision_base import GRID_SIDE
from solvers import get_game
from solvers.wend.vision import WendVisionParser

OUT = Path(__file__).resolve().parent.parent / "docs" / "gifs"
GAMES = ("queens", "zip", "tango", "mini-sudoku", "patches", "wend")

W, H = 960, 540
BOARD = 460
"""The board panel is a BOARD x BOARD square on the left."""
PAPER, INK, SOFT, FAINT, RULE = "#f6f5f1", "#16161a", "#5e5c56", "#9a978f", "#d8d5cc"
RED = (201, 42, 42)
FONTS = "/usr/share/fonts/opentype/inter/Inter-{}.otf"
MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"

DEDUCE_SECONDS = 9.0
"""Time budget for the deduction replay; each step gets an equal share within the bounds below."""
STEP_MS = (380, 1300)

INTRO = {
    "queens": ("Queens", "Exact cover with Dancing Links"),
    "zip": ("Zip", "Hamiltonian path search with A*"),
    "tango": ("Tango", "SAT, solved by MiniSat"),
    "mini-sudoku": ("Mini Sudoku", "SAT, explained with singles"),
    "patches": ("Patches", "Exact cover with Dancing Links"),
    "wend": ("Wend", "Trie search and exact tiling"),
}
READ = {
    "queens": "Each cell's LAB colour is clustered with K-Means into one region per queen (numbered).",
    "zip": "Hough circles find the numbered discs and OCR reads them; thickness probes find the walls (red).",
    "tango": "Sun or moon by colour, blue against red. Signs (red rings): an × has ink at its centre, an = a gap.",
    "mini-sudoku": "Cells holding ink are read by OCR; empty cells hold none at all.",
    "patches": "Badges are found by colour. A plain rectangle gives a shape; a + badge holds an area in white.",
    "wend": "Grey tiles are walls; letters are read by OCR on tight crops. Word lengths come from the slot rows.",
}


def font(size: int, weight: str = "Regular") -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONTS.format(weight), size)


def mono(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(MONO, size)


@dataclass
class Beat:
    """One frame of the story: what the board shows and what the caption says."""

    stage: int
    title: str
    text: str
    board: Image.Image
    ms: int
    counter: str = ""


@dataclass
class Replay:
    """What the board shows after each deduction step."""

    reveal: set[tuple[int, int]] = field(default_factory=set)
    focus: set[tuple[int, int]] = field(default_factory=set)
    crossed: set[tuple[int, int]] = field(default_factory=set)
    rejected: set[tuple[int, int]] = field(default_factory=set)


def cells_in(text: str) -> list[tuple[int, int]]:
    return [(int(r), int(c)) for r, c in re.findall(r"\((\d+), (\d+)\)", text)]


def rect_cells(top: int, left: int, height: int, width: int) -> set[tuple[int, int]]:
    return {(r, c) for r in range(top, top + height) for c in range(left, left + width)}


def patch_cells(message: str) -> set[tuple[int, int]]:
    match = re.search(r"(\d+)x(\d+) patch at \((\d+), (\d+)\)", message)
    if match is None:
        raise ValueError(f"no patch in {message!r}")
    height, width, top, left = map(int, match.groups())
    return rect_cells(top, left, height, width)


# --------------------------------------------------------------------------- per-game readouts


def readout(name: str, board: Any) -> tuple[dict[tuple[int, int], str], list[tuple[tuple[int, int], tuple[int, int]]]]:
    """Tags per cell, and cell pairs whose shared edge should be drawn as a found wall."""
    if name == "queens":
        return {(r, c): str(region + 1) for r, row in enumerate(board.regions) for c, region in enumerate(row)}, []
    if name == "zip":
        tags = {cell: str(i + 1) for i, cell in enumerate(board.checkpoints)}
        return tags, [tuple(sorted(wall)) for wall in board.walls]
    if name == "tango":
        return {cell: symbol.value[0].upper() for cell, symbol in board.givens.items()}, []
    if name == "mini-sudoku":
        return {cell: str(digit) for cell, digit in board.givens.items()}, []
    if name == "patches":
        return {clue.cell: clue.label for clue in board.clues}, []
    if name == "wend":
        return {(r, c): ch for r, row in enumerate(board.letters) for c, ch in enumerate(row) if ch}, []
    raise KeyError(name)


def replays(name: str, steps: list[Any], solution: Any) -> list[tuple[Any, Replay]]:
    """Board state after each shown step (Tango's givens are already on the screenshot, so they are skipped)."""
    state = Replay()
    out = []
    if name == "zip":
        path, shown = list(solution), 0
        for step in steps:  # grow the path three cells a frame, pausing on each checkpoint
            stop = path.index(step.cell) + 1
            for end in [*range(shown + 3, stop, 3), stop]:
                out.append((step, Replay(reveal=set(path[:end]), focus={path[end - 1]})))
            shown = stop
        return out
    for step in steps:
        if step.rule == "Given":
            continue
        state.focus = set()
        if name == "queens" and step.rule == "Single Choice":
            state.reveal.add(step.cell)
            state.focus = {step.cell}
        elif name == "queens":
            listed = set(cells_in(step.message.split("[", 1)[-1])) if "[" in step.message else set()
            state.crossed |= listed
            state.focus = listed
        elif name in ("tango", "mini-sudoku"):
            state.reveal.add(step.cell)
            state.focus = {step.cell}
        elif name == "patches" and step.rule == "Try and Fail":
            state.rejected = patch_cells(step.message)
            state.focus = set()
        elif name == "patches":
            state.rejected = set()
            cells = patch_cells(step.message)
            state.reveal |= cells
            state.focus = cells
        elif name == "wend":
            cells = set(cells_in(step.message.split(" because")[0]))
            state.reveal |= cells
            state.focus = cells
        out.append((step, replace(state, reveal=set(state.reveal), crossed=set(state.crossed))))
    return out


# --------------------------------------------------------------------------- drawing


def to_pil(bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(bgr[..., :3], cv2.COLOR_BGR2RGB))


def fit(image: Image.Image) -> Image.Image:
    """Letterbox ``image`` into the white BOARD square."""
    square = Image.new("RGB", (BOARD, BOARD), "white")
    scale = min(BOARD / image.width, BOARD / image.height)
    small = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))), Image.LANCZOS)
    square.paste(small, ((BOARD - small.width) // 2, (BOARD - small.height) // 2))
    return square


def cell_box(cell: tuple[int, int], n: int) -> tuple[int, int, int, int]:
    (x0, y0), (x1, y1) = visualizer.cell_rect(cell[0], cell[1], n, GRID_SIDE)
    return x0, y0, x1, y1


def with_layer(grid: np.ndarray, layer: np.ndarray, cells: set[tuple[int, int]], n: int) -> np.ndarray:
    """``grid`` with the solution layer shown only inside ``cells``."""
    mask = np.zeros(layer.shape[:2], dtype=bool)
    for cell in cells:
        x0, y0, x1, y1 = cell_box(cell, n)
        mask[y0:y1, x0:x1] = True
    alpha = (layer[..., 3:4].astype(np.float32) / 255) * mask[..., None]
    return (grid * (1 - alpha) + layer[..., :3] * alpha).astype(np.uint8)


def outline(draw: ImageDraw.ImageDraw, cells: set[tuple[int, int]], n: int, colour: Any, width: int) -> None:
    """Draw the outer boundary of a set of cells."""
    for r, c in cells:
        x0, y0, x1, y1 = cell_box((r, c), n)
        for (dr, dc), segment in (
            ((-1, 0), (x0, y0, x1, y0)),
            ((1, 0), (x0, y1, x1, y1)),
            ((0, -1), (x0, y0, x0, y1)),
            ((0, 1), (x1, y0, x1, y1)),
        ):
            if (r + dr, c + dc) not in cells:
                draw.line(segment, fill=colour, width=width)


def tag(draw: ImageDraw.ImageDraw, corner: tuple[float, float], text: str, size: int) -> None:
    """A small label in a cell's top-left corner, clear of what the cell shows."""
    face = mono(size)
    left, top, right, bottom = draw.textbbox((0, 0), text, font=face)
    x, y = corner
    draw.rectangle((x, y, x + right - left + 10, y + bottom - top + 9), fill="white", outline=INK, width=1)
    draw.text((x + 5 - left, y + 4 - top), text, font=face, fill=INK)


def board_frame(grid: np.ndarray, n: int, replay: Replay | None = None, layer: np.ndarray | None = None) -> Image.Image:
    shown = with_layer(grid, layer, replay.reveal, n) if replay and layer is not None else grid
    image = to_pil(shown)
    draw = ImageDraw.Draw(image)
    if replay:
        for r, c in replay.crossed - replay.reveal:
            x0, y0, x1, y1 = cell_box((r, c), n)
            pad = (x1 - x0) * 0.3
            draw.line((x0 + pad, y0 + pad, x1 - pad, y1 - pad), fill=RED, width=5)
            draw.line((x0 + pad, y1 - pad, x1 - pad, y0 + pad), fill=RED, width=5)
        if replay.rejected:
            outline(draw, replay.rejected, n, RED, 8)
        if replay.focus:
            outline(draw, replay.focus, n, "white", 12)
            outline(draw, replay.focus, n, INK, 6)
    return fit(image)


def compose(beat: Beat, intro: tuple[str, str], progress: float) -> Image.Image:
    frame = Image.new("RGB", (W, H), PAPER)
    frame.paste(beat.board, (40, 40))
    draw = ImageDraw.Draw(frame)
    draw.rectangle((39, 39, 40 + BOARD, 40 + BOARD), outline=RULE)
    x = 540
    title, subtitle = intro
    draw.text((x, 38), title, font=font(34, "Bold"), fill=INK)
    draw.text((x, 84), subtitle, font=font(17), fill=SOFT)
    stages = ("1 Find", "2 Grid", "3 Read", "4 Deduce", "5 Solved")
    sx = x
    for index, name in enumerate(stages, start=1):
        face = font(13, "SemiBold" if index == beat.stage else "Regular")
        colour = INK if index == beat.stage else FAINT
        draw.text((sx, 132), name, font=face, fill=colour)
        width = draw.textlength(name, font=face)
        if index == beat.stage:
            draw.line((sx, 152, sx + width, 152), fill=INK, width=2)
        sx += width + 22
    draw.line((x, 160, W - 40, 160), fill=RULE)
    y = 184
    if beat.counter:
        draw.text((x, y), beat.counter, font=mono(13), fill=FAINT)
        y += 24
    draw.text((x, y), beat.title, font=font(24, "SemiBold"), fill=INK)
    y += 42
    for line in wrap(draw, beat.text, font(17), W - 40 - x):
        draw.text((x, y), line, font=font(17), fill="#2c2b28")
        y += 26
    draw.text((x, H - 66), "real LinkedIn screenshot", font=mono(12), fill=FAINT)
    draw.line((x, H - 40, W - 40, H - 40), fill=RULE, width=3)
    draw.line((x, H - 40, x + (W - 40 - x) * progress, H - 40), fill=INK, width=3)
    return frame


def wrap(draw: ImageDraw.ImageDraw, text: str, face: ImageFont.FreeTypeFont, width: int) -> list[str]:
    lines, line = [], ""
    for word in text.split():
        candidate = f"{line} {word}".strip()
        if draw.textlength(candidate, font=face) <= width:
            line = candidate
        else:
            lines.append(line)
            line = word
    return [*lines, line][:9]


# --------------------------------------------------------------------------- the story


def story(name: str) -> tuple[list[Beat], tuple[str, str]]:
    game = get_game(name)
    if game.parser is None:
        raise ValueError(f"{name} has no vision stage")
    parser = game.parser()
    picture = real.image(name)
    grid = parser.rectify_grid(picture)
    n = parser.detect_grid_size(grid.image)
    board = parser.parse(picture)
    if name == "wend":
        board = replace(board, word_lengths=WendVisionParser().slot_lengths(real.image("wend-slots")))
    solver = game.solver()
    result = solver.run(board)
    latency = statistics.median(solver.run(board, track_memory=False).profile.latency_us for _ in range(5))
    parser.warp_to_image = lambda _image, canvas: canvas  # type: ignore[method-assign, assignment]  # keep the overlay in rectified coordinates
    layer = parser.overlay_layer(picture, board, result.solution)

    found = picture.copy()
    cv2.polylines(found, [grid.corners.astype(np.int32)], True, (26, 22, 22), max(3, picture.shape[0] // 160))
    for x, y in grid.corners.astype(int):
        cv2.circle(found, (int(x), int(y)), max(6, picture.shape[0] // 90), (42, 42, 201), -1)
    lines = grid.image.copy()
    step = GRID_SIDE / n
    for k in range(n + 1):
        cv2.line(lines, (round(k * step), 0), (round(k * step), GRID_SIDE), (42, 42, 201), 3)
        cv2.line(lines, (0, round(k * step)), (GRID_SIDE, round(k * step)), (42, 42, 201), 3)
    read = to_pil(grid.image)
    draw = ImageDraw.Draw(read)
    tags, walls = readout(name, board)
    for a, b in walls:
        (ax0, ay0, ax1, ay1), (bx0, by0, _, _) = cell_box(a, n), cell_box(b, n)
        segment = (bx0, ay0, bx0, ay1) if a[0] == b[0] else (ax0, by0, ax1, by0)
        draw.line(segment, fill=RED, width=12)
    for (r, c), text in tags.items():
        x0, y0, _, _ = cell_box((r, c), n)
        tag(draw, (x0 + 6, y0 + 6), text, 22 if n <= 7 else 18)
    for constraint in getattr(board, "constraints", ()):  # Tango signs: ring each one the parser found
        (ax0, ay0, ax1, ay1), (bx0, by0, bx1, by1) = cell_box(constraint.a, n), cell_box(constraint.b, n)
        cx, cy = (ax0 + ax1 + bx0 + bx1) / 4, (ay0 + ay1 + by0 + by1) / 4
        draw.ellipse((cx - 24, cy - 24, cx + 24, cy + 24), outline=RED, width=5)

    beats = [
        Beat(
            1,
            "Find the board",
            "Adaptive threshold, then the largest four-corner contour. A perspective warp straightens it to 720 × 720.",
            fit(to_pil(found)),
            1700,
        ),
        Beat(
            2,
            "Detect the grid",
            f"The edge profile is correlated with combs of evenly spaced lines. The best fit is {n} × {n}.",
            fit(to_pil(lines)),
            1500,
        ),
        Beat(3, "Read the cells", READ[name], fit(read), 2300),
    ]
    shown = replays(name, result.explanation.steps, result.solution)
    each = int(np.clip(DEDUCE_SECONDS * 1000 / max(1, len(shown)), *STEP_MS))
    rules = [s for s in result.explanation.steps if s.rule != "Given"]
    for step, replay in shown:
        number = rules.index(step) + 1
        message = step.message if len(step.message) < 230 else step.message[:227] + "…"
        beats.append(
            Beat(
                4, step.rule, message, board_frame(grid.image, n, replay, layer), each, f"step {number} / {len(rules)}"
            )
        )
    beats[-1].ms = max(beats[-1].ms, 1200)
    solved = type(parser)().generate_overlay(picture, board, result.solution)
    guesses = sum(s.rule in ("Search", "SAT") for s in result.explanation.steps)
    note = "no guessing" if not guesses else f"{guesses} search steps"
    beats.append(
        Beat(
            5,
            f"Solved in {latency / 1000:.1f} ms",
            f"{len(rules)} deduction steps, {note}. The answer is drawn back onto the original screenshot.",
            fit(to_pil(solved)),
            3200,
        )
    )
    return beats, INTRO[name]


def render(name: str) -> Path:
    beats, intro = story(name)
    total = sum(b.ms for b in beats)
    elapsed, frames = 0, []
    for beat in beats:
        frames.append(compose(beat, intro, elapsed / total))
        elapsed += beat.ms
    frames[-1] = compose(beats[-1], intro, 1.0)
    palette = Image.new("RGB", (W, H * len(frames)))
    for index, frame in enumerate(frames):
        palette.paste(frame, (0, H * index))
    palette = palette.resize((W // 2, H * len(frames) // 2)).quantize(colors=200, method=Image.Quantize.MEDIANCUT)
    quantized = [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in frames]
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.gif"
    quantized[0].save(
        path, save_all=True, append_images=quantized[1:], duration=[b.ms for b in beats], loop=0, optimize=True
    )
    return path


def main() -> None:
    for name in GAMES:
        path = render(name)
        print(f"{path.relative_to(OUT.parent.parent)}  {path.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
