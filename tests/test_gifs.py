"""The README animations: the replay must follow the solver's steps faithfully."""

from bench.gifs import patch_cells, replays
from core.explainer import DeductionStep


def test_zip_path_grows_to_each_checkpoint_and_ends_complete() -> None:
    path = [(0, 0), (0, 1), (0, 2), (1, 2), (1, 1), (1, 0)]
    steps = [DeductionStep(1, "Start", "", (0, 0)), DeductionStep(2, "Checkpoint", "", (1, 0))]
    frames = [replay.reveal for _, replay in replays("zip", steps, path)]
    assert all(a <= b for a, b in zip(frames, frames[1:]))  # never shrinks
    assert frames[0] == {(0, 0)} and frames[-1] == set(path)


def test_patches_try_and_fail_outlines_without_placing() -> None:
    placed = "clue 4 at (0, 0) takes the 2x2 patch at (0, 0) because it is the only rectangle left that fits"
    rejected = "clue 2 at (2, 0) cannot take the 1x2 patch at (2, 0): a few steps later a clue has no rectangle"
    steps = [DeductionStep(1, "Only Fit", placed, (0, 0)), DeductionStep(2, "Try and Fail", rejected, (2, 0))]
    (_, first), (_, second) = replays("patches", steps, [])
    assert first.reveal == patch_cells(placed) == {(0, 0), (0, 1), (1, 0), (1, 1)}
    assert second.reveal == first.reveal and second.rejected == {(2, 0), (2, 1)} and not second.focus
