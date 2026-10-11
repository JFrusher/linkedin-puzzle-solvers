# 0001: Solve with exact algorithms; explain by replaying human rules

**Context.** Each game needs both a fast, certain answer and a human-readable explanation. One option was to
make the deduction rules the solver, so the explanation is the solve.

**Decision.** Solvers use exact methods: Dancing Links for Queens and Patches, MiniSat for Tango and Mini
Sudoku, A* for Zip, a trie search for Wend. A separate `explain()` replays the known answer with human rules
outside the timed region, and labels a step `Search` or `SAT` only when no rule applies.

**Why.** Measured on Queens: running the rules to completion takes 1.1–2.9 ms, against 0.3–0.9 ms for Dancing
Links, which already expands about N nodes, leaving no search to prune. Keeping them apart also means a rule
bug cannot produce a wrong answer, only a wrong explanation, and the soundness tests catch those.

**Cost.** The rules live in a second code path, so every rule needs a soundness test against the true answer.
