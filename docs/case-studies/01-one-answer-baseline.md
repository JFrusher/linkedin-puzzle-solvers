# The baseline was lying

## Problem

Every solver can explain its answer with human rules ("Cell (2, 6) must be a Queen because Column 6 has no
other valid slot"), falling back to a `Search` step only when no rule applies. The first measurement on
generated Queens boards was bad: **54–72% of placements were `Search`**. Tango showed 16% `SAT` steps. The
obvious reaction would be to start writing more rules.

## Hypothesis

Real LinkedIn boards have exactly one answer. The board generators did not check that. On a board with several
answers, no sound rule can force a cell whose value differs between answers, so those boards can add `Search`
steps however good the rules are. The baseline was partly measuring the generator, not the rules.

## Measurement

The generators were changed to count answers (stopping at two) and keep only boards with exactly one: Queens
by reshaping regions with a hill climb, Tango and Zip by adding clues until unique, Wend by retrying.
`bench/deduction_report.py` then re-measured 20 boards per size.

| | Queens `Search` | Tango `SAT` |
|---|---|---|
| Before (any boards) | 54–72% | 16% |
| One-answer boards | 18–26% | 5% |

Most of the "missing rules" were never missing.

## Then the rules

With an honest baseline, rules were added one at a time and kept only if the numbers moved. For Queens, an
ablation over 80 boards (7×7 to 10×10, 680 placements) shows each rule earns its place:

| Rule set | `Search` placements | Try and Fail needed |
|---|---:|---:|
| All four rules | **0** / 680 | 2 |
| Without Try and Fail | 1 / 680 | 0 |
| Without Confinement | 0 / 680 | 40 |
| Without Shared Attack | 0 / 680 | 106 |
| Single Choice only | 146 / 680 | 0 |

Shared Attack and Confinement are what keep the expensive Try and Fail rule rare.

## Result

Queens, Tango, Wend, Mini Sudoku and Patches all explain 100% of their placements with rules. The rules do not
replace the solvers (see [decision 0001](../decisions/0001-solve-exactly-explain-by-replay.md)): Dancing Links
solves Queens in 0.3–0.9 ms, while running the rules to completion takes 1.1–2.9 ms.

**Lesson:** before optimising against a metric, check that the test data matches reality. Here one property of
the real game, a unique answer, changed the conclusion.
