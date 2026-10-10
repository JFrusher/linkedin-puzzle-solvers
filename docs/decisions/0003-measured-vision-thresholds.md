# 0003: Every vision threshold is measured on real and synthetic boards

**Context.** Vision runs on thresholds: how dark is ink, how thick is a wall, how coloured is a region.
Guessed thresholds passed every synthetic test and failed on five of six real screenshots.

**Decision.** A threshold is set only after measuring both classes it separates, on real captures
(`tests/fixtures/real/`) and synthetic renders, and its comment quotes those values. For example, Zip's wall
threshold of 0.075 sits between non-walls at 0.051 and the thinnest walls at 0.097.

**Why.** The comment shows how much margin there is, so a future failure can be judged against it rather than
rediscovered. See [case study 04](../case-studies/04-real-screenshots.md).

**Cost.** One real screenshot per game is a thin sample; more captures may move some thresholds.
