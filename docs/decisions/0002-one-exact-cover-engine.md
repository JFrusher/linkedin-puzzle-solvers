# 0002: One Dancing Links engine for every exact-cover game

**Context.** Queens was solved with Knuth's Algorithm X on Dancing Links. Patches, added later, is also exact
cover: every cell and every clue must be covered exactly once by the chosen rectangles.

**Decision.** Dancing Links moved from the Queens solver into `core/dlx.py`, with primary items (covered
exactly once) and secondary items (covered at most once). Queens uses secondary items for its "no two queens
touch" 2×2 windows; Patches uses cells plus one item per clue.

**Why.** One tested engine instead of two, and a new exact-cover game costs only its row construction. Patches
solves real boards in about 2 ms.

**Cost.** Both games depend on one module, so a change to `core/dlx.py` must pass both games' tests.
