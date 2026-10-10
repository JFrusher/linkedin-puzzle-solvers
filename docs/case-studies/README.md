# Case studies

Four problems from building this project, each written as problem, hypothesis, measurement and result. Every
number comes from a script in the repo (`python -m bench.deduction_report`, `python main.py benchmark`, or the
tests) and is recorded in [`../IMPROVEMENTS.md`](../IMPROVEMENTS.md).

| Study | Result |
|---|---|
| [The baseline was lying](01-one-answer-baseline.md) | Queens "guess" steps went from 54–72% to 18–26% by fixing the test boards alone, then to 0% with four rules, each proven necessary by ablation |
| [Making Zip's search fast where it was slow](02-zip-search.md) | Sparse 8×8 p90 from 5.3 s to 9 ms and the worst case from 18.3 s to 1.2 s; two plausible ideas measured and rejected |
| [Soundness tests catch rules that lie](03-tango-soundness.md) | Two subtle Tango rule bugs caught before shipping; 0 wrong deductions over 1,000 boards afterwards |
| [Calibrating vision on real screenshots](04-real-screenshots.md) | Six parsers that worked on synthetic boards met real captures; each failure measured, then fixed, ending 6/6 exact |

Design decisions behind these are in [`../decisions/`](../decisions/).
