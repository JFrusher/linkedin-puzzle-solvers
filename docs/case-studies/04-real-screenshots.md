# Calibrating vision on real screenshots

## Problem

All six screenshot parsers passed their tests on synthetic boards rendered by the project's own generator.
The first real LinkedIn captures were the real test.

| Game | First run on the real screenshot |
|---|---|
| Queens | correct |
| Zip | one wall too many |
| Mini Sudoku | missed every "1" |
| Tango | every sun and moon swapped, every sign missed |
| Patches | crashed |
| Wend | crashed |

## Method

For each failure: state a theory, measure it on the real and synthetic images, and change code only once the
measurement proves the theory. Each threshold in the code now quotes the values it separates.

| Game | Measured cause | Fix |
|---|---|---|
| Zip | the "1" disc reaches the wall probe: 0.051 of a cell against a 0.05 threshold; real walls measure 0.194, synthetic 0.097 | threshold 0.075 |
| Mini Sudoku | a thin "1" has 0.016 ink against a 0.02 threshold; empty cells measure exactly 0 | threshold 0.005 |
| Tango | grey templates of the project's own icons matched LinkedIn's badly; LinkedIn's signs are no darker than grey 117, the ink test wanted under 100 | symbols by colour (blue minus red: moons +145, suns −204); signs by ink at the sign's centre: × 1.0, = 0.0 |
| Patches | the clue layout had been guessed: real clues are coloured badges holding an area or a shape | badges by saturation; fill 0.98+ is a plain shape, 0.79–0.85 a "+" badge; white holes are digits |
| Wend | walls are light grey (178), not dark; loose crops read U as "PU" (36/40 tiles); WordNet has DOWNLOAD but not DOWNLOADED | median grey under 215; tight crops (40/40); generated inflections |

Two theories were wrong and were dropped on the evidence: Tango's last misread sign was not caused by grid
lines, and Wend's "PU" was not caused by wall borders in the crop (the crop was clean). In both cases the
measurement pointed elsewhere.

## Result

All six games read their real screenshot exactly and solve it, the synthetic renders were moved toward the
real look so their tests check the same features, and `tests/test_real.py` keeps it that way. The same
evidence later made automatic game detection work: Queens boards are 66%+ coloured against 21% at most for
any other game, and Tango's colour is only orange and blue, so each reader rejects the others' boards.

**Lesson:** synthetic data proves the pipeline works on what you imagined. One real example per case
disproved assumptions in five of six parsers, and every fix was a measured threshold, not a guess.
