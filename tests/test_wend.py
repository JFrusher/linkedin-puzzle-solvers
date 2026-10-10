"""Wend: Trie-guided DFS tiling with island pruning and OCR vision."""

import pytest

from bench.synthetic import WEND_SAMPLE, render_wend
from core.nlp import dictionary_words
from core.solver_base import UnsolvableError
from solvers.wend.solver import MIN_WORD_LENGTH, Trie, WendBoard, WendSolver, is_valid_solution
from solvers.wend.vision import WendVisionParser


def test_trie_marks_words_and_prefixes() -> None:
    trie = Trie(["CAT", "CATS"])
    node = trie.root.children["C"].children["A"].children["T"]
    assert node.is_word and node.children["S"].is_word
    assert not trie.root.children["C"].is_word


def test_small_dictionary_tiling() -> None:
    board = WendBoard((("C", "A", "T"), ("D", "O", "G"), ("O", "W", "L")))
    solution = WendSolver(["CAT", "DOG", "OWL"]).solve(board)
    assert sorted(p.word for p in solution) == ["CAT", "DOG", "OWL"]


def test_island_pruning_rejects_untileable_board() -> None:
    board = WendBoard((("C", "A", "T"), ("X", None, None), (None, None, None)))
    with pytest.raises(UnsolvableError):
        WendSolver(["CAT"]).solve(board)


def test_word_lengths_must_cover_board() -> None:
    with pytest.raises(ValueError):
        WendBoard((("C", "A", "T"),) * 3, word_lengths=(3,))


def test_sample_with_wordnet_dictionary() -> None:
    result = WendSolver().run(WEND_SAMPLE)
    assert is_valid_solution(WEND_SAMPLE, result.solution, dictionary_words(MIN_WORD_LENGTH))


def test_vision_reads_letters_walls_and_slots() -> None:
    assert WendVisionParser().parse(render_wend(WEND_SAMPLE)) == WEND_SAMPLE
