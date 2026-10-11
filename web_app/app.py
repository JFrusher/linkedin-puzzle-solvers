"""Streamlit playground: upload or generate a board, inspect each vision stage and the solver run.

streamlit run web_app/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # streamlit runs this file as a script

from bench import synthetic  # noqa: E402
from core import visualizer  # noqa: E402
from core.solver_base import UnsolvableError  # noqa: E402
from core.vision_base import GridNotFoundError  # noqa: E402
from main import solve_image  # noqa: E402
from solvers import GAMES, get_game  # noqa: E402

PASTE = "Paste a screenshot (auto-detect)"
IMAGE_TYPES = ["png", "jpg", "jpeg"]
EXAMPLES = Path(__file__).resolve().parents[1] / "examples"  # typed example boards, one per typeable game

SAMPLES = {
    "queens": lambda seed: synthetic.render_queens(synthetic.generate_queens(8, seed)[0], seed),
    "zip": lambda seed: synthetic.render_zip(synthetic.generate_zip(6, seed)[0], seed),
    "tango": lambda seed: synthetic.render_tango(synthetic.generate_tango(6, seed)[0], seed),
    "wend": lambda seed: synthetic.render_wend(synthetic.WEND_SAMPLE, seed),
    "crossclimb": lambda seed: synthetic.render_crossclimb(synthetic.CROSSCLIMB_SAMPLE),
    "mini-sudoku": lambda seed: synthetic.render_sudoku(synthetic.generate_sudoku(seed)[0], seed),
    "patches": lambda seed: synthetic.render_patches(synthetic.generate_patches(6, seed)[0], seed),
}


def rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def show_run(profile, steps) -> None:  # type: ignore[no-untyped-def]
    left, middle, right = st.columns(3)
    left.metric("Solve latency", f"{profile.latency_us:,.0f} µs")
    middle.metric("Peak memory", f"{profile.peak_memory_kb:,.1f} KiB")
    right.metric("Nodes expanded", f"{profile.nodes_expanded:,}")
    st.subheader("Deduction log")
    st.dataframe(
        pd.DataFrame([{"Step": s.index, "Rule": s.rule, "Cell": s.cell, "Deduction": s.message} for s in steps]),
        hide_index=True,
        use_container_width=True,
    )


def pinpoint_page() -> None:
    from solvers.pinpoint.semantic_engine import PinpointSolver

    text = st.text_area("Clues, one per line", "pawn\nrook\nknight")
    clues = [line.strip() for line in text.splitlines() if line.strip()]
    if st.button("Rank categories", type="primary") and clues:
        result = PinpointSolver().run(clues)
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Category": g.label,
                        "Confidence %": round(g.confidence, 1),
                        "Coverage": g.coverage,
                        "Cosine": g.cosine,
                    }
                    for g in result.solution
                ]
            ),
            hide_index=True,
            use_container_width=True,
        )
        show_run(result.profile, result.explanation.steps)


def paste_page() -> None:
    st.markdown(
        "Take a screenshot of the board (**Win+Shift+S**), click the box below, press **Ctrl+V**, then **Enter**. "
        "The game is detected for you."
    )
    with st.container():  # keeps the paste box in the page instead of pinned to the bottom
        pasted = st.chat_input("Click here, Ctrl+V a screenshot, Enter", accept_file=True, file_type=IMAGE_TYPES)
    if not pasted or not pasted.files:
        return
    image = cv2.imdecode(np.frombuffer(pasted.files[0].getvalue(), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        st.error("That paste is not a decodable image.")
        return
    try:
        with st.spinner("Detecting the game and solving…"):
            outcome = solve_image(None, image)
    except (UnsolvableError, ValueError) as error:
        st.error(f"{type(error).__name__}: {error}")
        st.image(rgb(image), caption="Pasted")
        return
    solved = visualizer.composite(image, outcome["layer"])
    st.subheader(GAMES[outcome["game"]].title)
    if outcome["difficulty"] is not None:
        st.caption(f"Difficulty: {outcome['difficulty']}")
    st.image(rgb(solved), caption="Right-click → Copy image, then Ctrl+V it anywhere", width=640)
    st.download_button("Download PNG", cv2.imencode(".png", solved)[1].tobytes(), "solved.png", "image/png")
    show_run(outcome["profile"], outcome["steps"])


def typed_page(name: str) -> None:
    game = get_game(name)
    assert game.text is not None
    example = (EXAMPLES / f"{name}.txt").read_text()
    help_text = {
        "wend": "One row of letters per line, # for a wall; optionally a last line of word lengths.",
        "crossclimb": "The word length, then the clue shared by the top and bottom words, then the middle clues.",
    }[name]
    with st.form(f"typed-{name}"):
        text = st.text_area("Board", example, height=260, help=help_text)
        submitted = st.form_submit_button("Solve", type="primary")
    if not submitted:
        return
    try:
        board = game.text(text)
        result = game.solver().run(board)
    except (UnsolvableError, ValueError) as error:
        st.error(f"{type(error).__name__}: {error}")
        return
    st.success("Answer: " + ", ".join(item.word for item in result.solution))
    show_run(result.profile, result.explanation.steps)


def vision_page(name: str) -> None:
    game = get_game(name)
    assert game.parser is not None
    sources = ["Synthetic sample", "Upload screenshot"] + (["Type the board"] if game.text else [])
    source = st.radio("Board source", sources, horizontal=True)
    if source == "Type the board":
        typed_page(name)
        return
    if source == "Upload screenshot":
        upload = st.file_uploader("Screenshot", type=IMAGE_TYPES)
        if upload is None:
            return
        image = cv2.imdecode(np.frombuffer(upload.getvalue(), dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            st.error("That file is not a decodable image.")
            return
    else:
        image = SAMPLES[name](st.number_input("Seed", min_value=0, value=1, step=1))

    parser = game.parser()
    try:
        board = parser.parse(image)
        result = game.solver().run(board)
    except (GridNotFoundError, UnsolvableError, ValueError) as error:
        st.error(f"{type(error).__name__}: {error}")
        st.image(rgb(image), caption="Input")
        return

    tabs = st.tabs(["Solution", "Raw", "Contours", "HSV Masks", "Graph Nodes", "Board model"])
    tabs[0].image(rgb(parser.generate_overlay(image, board, result.solution)), use_container_width=True)
    if name == "crossclimb":
        # Crossclimb is a list of rows, not a square grid, so the grid debug views do not apply.
        tabs[1].image(rgb(image), use_container_width=True)
        for tab in tabs[2:5]:
            tab.info("Grid debug views apply to square-grid games only.")
    else:
        for tab, (title, view) in zip(tabs[1:5], parser.debug_views(image).items(), strict=True):
            tab.image(rgb(view), caption=title, use_container_width=True)
    tabs[5].write(board)
    show_run(result.profile, result.explanation.steps)


def main() -> None:
    st.set_page_config(page_title="Puzzle Solvers Playground", layout="wide")
    st.title("LinkedIn Puzzle Solvers — playground")
    name = st.sidebar.selectbox(
        "Game", [PASTE, *GAMES], format_func=lambda key: key if key == PASTE else GAMES[key].title
    )
    if name == PASTE:
        paste_page()
    elif name == "pinpoint":
        pinpoint_page()
    else:
        vision_page(name)


main()
