# Soundness tests catch rules that lie

## Problem

A deduction rule that is wrong is worse than no rule: it explains a step with confident, false reasoning, and
in a solver it can prune the real answer. Tango needed new rules to remove its last `SAT` (guess) steps, and
the sign rules interact in ways that are easy to get subtly wrong.

## Method

Every rule runs under a soundness test: generate one-answer boards, replay the explainer, and fail if any
deduction contradicts the known answer. Rules were then added one at a time over 40 one-answer boards.

| Rules | `SAT` steps | Try and Fail steps |
|---|---:|---:|
| Relation, No Three, Balance | 72 | 0 |
| + Sign Balance | 30 | 0 |
| + Equal Pair Neighbour | 15 | 0 |
| + Try and Fail | **0** | 18 |

## What the test caught

Sign Balance counts how many suns and moons the signs force into a line. The soundness test failed on two
cases before the rule shipped:

- **Chains.** For a×b×c, the first draft counted "one of each" per pair, two pairs, two of each. In fact a and c
  are equal, so the chain holds two of one symbol and one of the other.
- **Overlapping pairs.** An = pair that shares a cell with an already-counted × pair needs only one more slot,
  not two.

Both produced deductions that looked reasonable in the log and were wrong on a few boards in a thousand.

## Result

After the fixes: **0 wrong deductions over 1,000 boards**, and the rules alone fill every cell, so `SAT` steps
went from 72 to 0. The same soundness pattern now guards Queens, Wend and Patches (a Try and Fail rectangle may
never be part of the answer).

**Lesson:** for logic that explains itself, test against ground truth on many generated cases, not on
hand-picked examples. The bugs here only showed up a few times per thousand boards.
