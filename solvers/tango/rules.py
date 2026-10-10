"""Human-style Tango deductions; each rule returns the first cell it can fix, or ``None``.

* **Relation** — a sign next to a known cell fixes its partner.
* **No Three** — two equal cells in a window of three force the third to the other symbol.
* **Balance** — a line already holding N/2 of one symbol fills the rest with the other.
* **Sign Balance** — unknown × pairs inside a line that share no cell each hold one of each symbol, so
  they count towards both totals early; an unknown = pair adds two of one symbol, so it cannot take a
  symbol the line has fewer than two of left.
* **Equal Pair Neighbour** — a known cell right next to an unknown = pair along a line forces the pair
  to the other symbol, or the three would match.
* **Try and Fail** — giving a cell one symbol and following the rules above breaks a rule, so the cell
  takes the other symbol.
"""

from __future__ import annotations

from collections.abc import Callable

from solvers.tango.solver import Cell, Constraint, Relation, Symbol, TangoBoard

Known = dict[Cell, Symbol]
Deduction = tuple[Cell, Symbol, str, str]


def lines(n: int) -> list[list[Cell]]:
    return [[(i, j) for j in range(n)] for i in range(n)] + [[(j, i) for j in range(n)] for i in range(n)]


def relation(board: TangoBoard, known: Known) -> Deduction | None:
    for constraint in board.constraints:
        for src, dst in ((constraint.a, constraint.b), (constraint.b, constraint.a)):
            if src in known and dst not in known:
                symbol = known[src] if constraint.relation is Relation.EQUAL else known[src].other
                return dst, symbol, "Relation", f"of the '{constraint.relation.value}' sign with {src}"
    return None


def no_three(board: TangoBoard, known: Known) -> Deduction | None:
    for line in lines(board.size):
        for k in range(board.size - 2):
            a, b, c = line[k : k + 3]
            for x, y, target in ((a, b, c), (b, c, a), (a, c, b)):
                if x in known and y in known and target not in known and known[x] == known[y]:
                    return target, known[x].other, "No Three", f"{x} and {y} are both {known[x].value}"
    return None


def balance(board: TangoBoard, known: Known) -> Deduction | None:
    half = board.size // 2
    for line in lines(board.size):
        for symbol in Symbol:
            if sum(known.get(cell) == symbol for cell in line) == half:
                for cell in line:
                    if cell not in known:
                        return cell, symbol.other, "Balance", f"its line already holds {half} {symbol.value}s"
    return None


def _unknown_pairs(board: TangoBoard, known: Known, line: list[Cell], sign: Relation) -> list[Constraint]:
    members = set(line)
    return [
        c
        for c in board.constraints
        if c.relation is sign and c.a in members and c.b in members and c.a not in known and c.b not in known
    ]


def sign_balance(board: TangoBoard, known: Known) -> Deduction | None:
    half = board.size // 2
    for line in lines(board.size):
        # Only pairs sharing no cell hold exactly one of each: a chain a×b×c holds two of one symbol.
        opposite: list[Constraint] = []
        paired: set[Cell] = set()
        for pair in _unknown_pairs(board, known, line, Relation.OPPOSITE):
            if pair.a not in paired and pair.b not in paired:
                opposite.append(pair)
                paired |= {pair.a, pair.b}
        equal = _unknown_pairs(board, known, line, Relation.EQUAL)
        for symbol in Symbol:
            have = sum(known.get(cell) == symbol for cell in line) + len(opposite)
            if opposite and have == half:
                for cell in line:
                    if cell not in known and cell not in paired:
                        return (
                            cell,
                            symbol.other,
                            "Sign Balance",
                            f"its line holds {have - len(opposite)} {symbol.value}s plus one in each × pair, "
                            f"which makes {half}",
                        )
            if half - have < 2:
                for pair in equal:
                    if pair.a not in paired and pair.b not in paired:
                        return (
                            pair.a,
                            symbol.other,
                            "Sign Balance",
                            f"its = pair with {pair.b} needs two matching symbols but the line has room for "
                            f"{half - have} more {symbol.value}",
                        )
    return None


def equal_pair_neighbour(board: TangoBoard, known: Known) -> Deduction | None:
    for line in lines(board.size):
        for pair in _unknown_pairs(board, known, line, Relation.EQUAL):
            i, j = sorted((line.index(pair.a), line.index(pair.b)))
            for k in (i - 1, j + 1):
                if 0 <= k < board.size and line[k] in known:
                    symbol = known[line[k]]
                    return (
                        pair.a,
                        symbol.other,
                        "Equal Pair Neighbour",
                        f"its = pair with {pair.b} sits next to {line[k]} ({symbol.value}); matching would make three",
                    )
    return None


def broken(board: TangoBoard, known: Known) -> bool:
    """True when the partial grid already breaks a rule."""
    half = board.size // 2
    for line in lines(board.size):
        values = [known.get(cell) for cell in line]
        for symbol in {v for v in values if v is not None}:
            if values.count(symbol) > half:
                return True
        if any(values[k] is not None and values[k] == values[k + 1] == values[k + 2] for k in range(board.size - 2)):
            return True
    return any(
        c.a in known and c.b in known and (known[c.a] == known[c.b]) != (c.relation is Relation.EQUAL)
        for c in board.constraints
    )


SIMPLE: tuple[Callable[[TangoBoard, Known], Deduction | None], ...] = (
    relation,
    no_three,
    balance,
    sign_balance,
    equal_pair_neighbour,
)


def try_and_fail(board: TangoBoard, known: Known) -> Deduction | None:
    n = board.size
    for cell in [(r, c) for r in range(n) for c in range(n) if (r, c) not in known]:
        for symbol in Symbol:
            trial = dict(known)
            trial[cell] = symbol
            while not broken(board, trial) and len(trial) < n * n:
                step = next((d for rule in SIMPLE if (d := rule(board, trial)) is not None), None)
                if step is None:
                    break
                trial[step[0]] = step[1]
            if broken(board, trial):
                return cell, symbol.other, "Try and Fail", f"making it {symbol.value} breaks a rule a few steps later"
    return None


RULES: tuple[Callable[[TangoBoard, Known], Deduction | None], ...] = (*SIMPLE, try_and_fail)


def next_deduction(board: TangoBoard, known: Known) -> Deduction | None:
    for rule in RULES:
        deduction = rule(board, known)
        if deduction is not None:
            return deduction
    return None
