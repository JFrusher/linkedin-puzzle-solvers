# 0005: Detect the game by trying each reader, with rejection checks

**Context.** Pasting a screenshot should not need a game picker. A trained classifier would need labelled
data the project does not have.

**Decision.** Try the readers in a fixed order (Queens, Zip, Tango, Mini Sudoku, Wend, Patches) and keep the
first that both reads and solves the board. Readers reject boards that are not theirs: Queens needs at least
40% coloured regions, and Tango allows at most 10% colour that is not sun orange or moon blue. Patches goes
last, because its badge reader also finds the colours of Queens and Tango boards.

**Why.** Solving is a strong check: a wrong reader rarely produces a board that also has an answer. The two
rejection checks fixed the two confusions measured on 37 boards (a Patches board read as Tango, a Wend board
read as Queens); detection is now 37/37.

**Cost.** Detection can take a few seconds the first time, while the OCR model loads.
