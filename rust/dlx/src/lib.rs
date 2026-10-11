//! Knuth's Algorithm X on Dancing Links: a Rust twin of `core/dlx.py`.
//!
//! Same API and the same search order (minimum-remaining-values column, first column wins ties, rows top to
//! bottom), so it yields the same covers in the same order and counts the same `nodes_expanded`. Nodes live
//! in flat index arrays instead of linked objects, and the search is an explicit stack, so `search()` can
//! yield covers lazily like the Python generator.

use pyo3::exceptions::PyIndexError;
use pyo3::prelude::*;

const ROOT: usize = 0;

/// One level of the search: the column branched on and the row currently chosen for it.
struct Frame {
    column: usize,
    row: usize,
}

#[derive(PartialEq)]
enum Mode {
    Descend,
    Advance,
    Finished,
}

#[pyclass(module = "dlx_rs")]
struct DancingLinks {
    left: Vec<usize>,
    right: Vec<usize>,
    up: Vec<usize>,
    down: Vec<usize>,
    column: Vec<usize>,
    row_id: Vec<i64>,
    size: Vec<usize>,
    columns: usize,
    stack: Vec<Frame>,
    mode: Mode,
    #[pyo3(get)]
    nodes_expanded: u64,
}

impl DancingLinks {
    fn push_node(&mut self, column: usize, row_id: i64) -> usize {
        let node = self.left.len();
        self.left.push(node);
        self.right.push(node);
        self.up.push(node);
        self.down.push(node);
        self.column.push(column);
        self.row_id.push(row_id);
        self.size.push(0);
        node
    }

    fn cover(&mut self, c: usize) {
        let (l, r) = (self.left[c], self.right[c]);
        self.left[r] = l;
        self.right[l] = r;
        let mut row = self.down[c];
        while row != c {
            let mut node = self.right[row];
            while node != row {
                let (u, d) = (self.up[node], self.down[node]);
                self.up[d] = u;
                self.down[u] = d;
                self.size[self.column[node]] -= 1;
                node = self.right[node];
            }
            row = self.down[row];
        }
    }

    fn uncover(&mut self, c: usize) {
        let mut row = self.up[c];
        while row != c {
            let mut node = self.left[row];
            while node != row {
                self.size[self.column[node]] += 1;
                let (u, d) = (self.up[node], self.down[node]);
                self.up[d] = node;
                self.down[u] = node;
                node = self.left[node];
            }
            row = self.up[row];
        }
        let (l, r) = (self.left[c], self.right[c]);
        self.left[r] = c;
        self.right[l] = c;
    }

    /// Cover every other column of a chosen row.
    fn choose(&mut self, row: usize) {
        self.nodes_expanded += 1;
        let mut node = self.right[row];
        while node != row {
            self.cover(self.column[node]);
            node = self.right[node];
        }
    }

    fn unchoose(&mut self, row: usize) {
        let mut node = self.left[row];
        while node != row {
            self.uncover(self.column[node]);
            node = self.left[node];
        }
    }

    /// Minimum-remaining-values: the first primary column with the fewest rows.
    fn tightest_column(&self) -> usize {
        let mut best = self.right[ROOT];
        let mut c = self.right[best];
        while c != ROOT {
            if self.size[c] < self.size[best] {
                best = c;
            }
            c = self.right[c];
        }
        best
    }

    /// Advance to the next exact cover, or `None` when the search is exhausted.
    fn next_cover(&mut self) -> Option<Vec<i64>> {
        loop {
            match self.mode {
                Mode::Finished => return None,
                Mode::Descend => {
                    if self.right[ROOT] == ROOT {
                        self.mode = Mode::Advance;
                        return Some(self.stack.iter().map(|f| self.row_id[f.row]).collect());
                    }
                    let c = self.tightest_column();
                    if self.size[c] == 0 {
                        self.mode = Mode::Advance;
                        continue;
                    }
                    self.cover(c);
                    let row = self.down[c];
                    self.stack.push(Frame { column: c, row });
                    self.choose(row);
                }
                Mode::Advance => {
                    let Some(top) = self.stack.last() else {
                        self.mode = Mode::Finished;
                        continue;
                    };
                    let (c, row) = (top.column, top.row);
                    self.unchoose(row);
                    let next = self.down[row];
                    if next == c {
                        self.uncover(c);
                        self.stack.pop();
                        continue;
                    }
                    self.stack.last_mut().expect("frame exists").row = next;
                    self.choose(next);
                    self.mode = Mode::Descend;
                }
            }
        }
    }
}

#[pymethods]
impl DancingLinks {
    /// Columns `0..primary-1` must be covered exactly once, the next `secondary` at most once.
    #[new]
    fn new(primary: usize, secondary: usize) -> Self {
        let mut links = DancingLinks {
            left: Vec::new(),
            right: Vec::new(),
            up: Vec::new(),
            down: Vec::new(),
            column: Vec::new(),
            row_id: Vec::new(),
            size: Vec::new(),
            columns: primary + secondary,
            stack: Vec::new(),
            mode: Mode::Descend,
            nodes_expanded: 0,
        };
        links.push_node(ROOT, -1);
        for index in 0..primary + secondary {
            let header = links.push_node(index + 1, -1);
            links.column[header] = header;
            if index < primary {
                // Primary headers join the root list; secondary ones stay out, so search never requires them.
                let last = links.left[ROOT];
                links.left[header] = last;
                links.right[header] = ROOT;
                links.right[last] = header;
                links.left[ROOT] = header;
            }
        }
        links
    }

    /// Append a matrix row with 1s in `column_ids`.
    fn add_row(&mut self, row_id: i64, column_ids: Vec<usize>) -> PyResult<()> {
        if let Some(bad) = column_ids.iter().find(|&&c| c >= self.columns) {
            return Err(PyIndexError::new_err(format!(
                "column {bad} is out of range for {} columns",
                self.columns
            )));
        }
        let mut first: Option<usize> = None;
        for column_id in column_ids {
            let header = column_id + 1;
            let node = self.push_node(header, row_id);
            let above = self.up[header];
            self.up[node] = above;
            self.down[node] = header;
            self.down[above] = node;
            self.up[header] = node;
            self.size[header] += 1;
            match first {
                None => first = Some(node),
                Some(f) => {
                    let last = self.left[f];
                    self.left[node] = last;
                    self.right[node] = f;
                    self.right[last] = node;
                    self.left[f] = node;
                }
            }
        }
        Ok(())
    }

    /// Iterate over every exact cover as a list of row ids.
    fn search(mut slf: PyRefMut<'_, Self>) -> PyRefMut<'_, Self> {
        // Unwind a partly consumed search so the new one starts from the full matrix.
        while let Some(frame) = slf.stack.pop() {
            slf.unchoose(frame.row);
            slf.uncover(frame.column);
        }
        slf.mode = Mode::Descend;
        slf
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self) -> Option<Vec<i64>> {
        self.next_cover()
    }
}

#[pymodule]
fn dlx_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<DancingLinks>()?;
    Ok(())
}
