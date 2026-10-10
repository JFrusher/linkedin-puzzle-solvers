# dlx_rs

A Rust twin of [`core/dlx.py`](../../core/dlx.py): Knuth's Algorithm X on Dancing Links, exposed to Python with
PyO3. Same API (`DancingLinks(primary, secondary)`, `add_row`, `search()`, `nodes_expanded`) and the same search
order, so it yields identical covers with identical node counts. See the repository README for the benchmark.

```bash
pip install ./rust/dlx           # builds with maturin; needs a Rust toolchain
python -m bench.dlx_speed        # Python vs Rust on the same rows
pytest tests/test_dlx_rust.py    # equivalence checks
```
