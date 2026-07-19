"""Load and resolve portfolio specs (portfolio.yaml). v0.1.0 -- implemented.

Notes
-----
What this file does: turns a YAML portfolio spec into validated, immutable
dataclasses (load_portfolio), then converts holdings into a single weight
vector at as-of prices (resolve_weights). Everything downstream -- stats,
decompose, optimize, stress -- consumes the resolved weights; no other
module re-derives them.

Spec rules, enforced here and nowhere else:
- every position has `shares` OR `weight`, never both, never neither
- shares > 0 and finite; fractional shares are fine
- weight in (0, 1]; explicit weights sum to <= 1 (float tolerance)
- symbols are uppercased on load; duplicates after normalization are an
  error (brk-b and BRK-B are the same holding)
- cash >= 0 and finite; a portfolio with no positions is legal only if it
  holds cash
- unknown YAML keys are errors, not warnings: a typo like `wieght:` must
  fail loudly or it silently changes the portfolio
- numbers must be YAML numbers; "80" (a string) is rejected, never coerced

The only non-obvious math here is mixing shares with weights. Explicit
weight positions claim a fraction W of total NAV N. Share positions are
worth D dollars at as-of prices, cash is C. The dollar legs must occupy
whatever fraction the weights left over:

    D + C = (1 - W) * N        =>        N = (D + C) / (1 - W)

Then share position i gets weight shares_i * price_i / N, explicit weights
pass through, cash gets C / N, and everything sums to 1. Two consequences
fall out of the algebra:
- W == 1 alongside share positions or cash is contradictory (no finite N)
  and raises
- a spec with ONLY weight positions and zero cash has no dollar anchor, so
  NAV is undefined (None) and weights pass through scale-free, remainder
  reported as cash weight

Failure style: any violation raises PortfolioSpecError naming the file,
position, and field. Nothing is repaired silently.
"""

import math
from collections.abc import Mapping  # accept any dict-like for prices, not just dict
from dataclasses import dataclass, field
from pathlib import Path

import yaml

# Public API of this module; `from ballast.portfolio.spec import *` honors it,
# and it doubles as a table of contents.
__all__ = [
    "PortfolioSpecError",
    "Position",
    "Portfolio",
    "ResolvedPortfolio",
    "load_portfolio",
    "resolve_weights",
]

# Tolerance for float comparisons on weight sums: [0.3, 0.3, 0.4] sums to
# 0.9999999999999999 in binary floating point and must count as 1.0.
_TOL = 1e-9

# The complete YAML vocabulary. Anything outside these sets is a typo by
# definition, and typos are errors (see module notes).
_ALLOWED_ROOT_KEYS = {"name", "positions", "cash"}
_ALLOWED_POSITION_KEYS = {"symbol", "shares", "weight"}


class PortfolioSpecError(ValueError):
    """Raised for any invalid portfolio spec. Message names the offending field.

    Subclasses ValueError so callers who don't know about Ballast's error
    types still catch it where they'd catch a ValueError.
    """


def _require_number(value: object, where: str) -> float:
    """Validate that a parsed value is a real, finite number.

    bool is checked first because Python's bool subclasses int, so
    `isinstance(True, int)` is True and `shares: true` would otherwise
    slip through as 1.0.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        # `where` is supplied by the caller so this message can say exactly
        # which field of which position is wrong.
        raise PortfolioSpecError(f"{where} must be a number, got {value!r}")
    out = float(value)  # normalize ints to float so downstream math is uniform
    if not math.isfinite(out):
        # Catches NaN and +/-inf. YAML happily parses `.nan` and `.inf`,
        # and NaN is especially nasty: it makes every comparison False.
        raise PortfolioSpecError(f"{where} must be finite, got {value!r}")
    return out


# frozen=True: instances are immutable -- a validated Position can never drift
# into an invalid state later. slots=True: fixed attribute set, less memory,
# and typos like pos.shres raise instead of silently creating an attribute.
@dataclass(frozen=True, slots=True)
class Position:
    """One holding: a symbol with shares (a quantity) or weight (a fraction of NAV)."""

    symbol: str
    shares: float | None = None  # exactly one of shares/weight is set...
    weight: float | None = None  # ...enforced in __post_init__ below

    def __post_init__(self) -> None:
        # __post_init__ runs automatically right after dataclass field
        # assignment. Validation lives here so an invalid Position cannot be
        # constructed at all -- not from YAML, not programmatically in tests.
        if not isinstance(self.symbol, str) or not self.symbol.strip():
            raise PortfolioSpecError(
                f"position symbol must be a non-empty string, got {self.symbol!r}"
            )
        symbol = self.symbol.strip().upper()  # "nvda " -> "NVDA"
        if any(ch.isspace() for ch in symbol):
            # Internal whitespace ("BRK B") is a malformed ticker, not two tickers.
            raise PortfolioSpecError(f"position symbol may not contain whitespace: {symbol!r}")
        # Frozen dataclass: plain `self.symbol = ...` raises FrozenInstanceError,
        # so the normalized value goes in through object.__setattr__. This is
        # the standard idiom for normalizing fields on frozen dataclasses.
        object.__setattr__(self, "symbol", symbol)

        # XOR check: (None == None) and (set == set) are both wrong states.
        # "both None" and "both set" each make this condition True.
        if (self.shares is None) == (self.weight is None):
            raise PortfolioSpecError(
                f"position {symbol!r} must have exactly one of `shares` or `weight`"
            )
        if self.shares is not None:
            shares = _require_number(self.shares, f"position {symbol!r}: shares")
            if shares <= 0:
                # Zero shares is a position that doesn't exist; negative is a
                # short, which the spec doesn't support (yet -- see IDEAS.md).
                raise PortfolioSpecError(f"position {symbol!r}: shares must be > 0, got {shares}")
            object.__setattr__(self, "shares", shares)
        if self.weight is not None:
            weight = _require_number(self.weight, f"position {symbol!r}: weight")
            if not 0 < weight <= 1:
                raise PortfolioSpecError(
                    f"position {symbol!r}: weight must be in (0, 1], got {weight}"
                )
            object.__setattr__(self, "weight", weight)


@dataclass(frozen=True, slots=True)
class Portfolio:
    """A validated spec: named positions plus cash. Immutable once constructed."""

    name: str
    # A tuple, not a list: lists are mutable, and a frozen dataclass holding a
    # list would be frozen in name only.
    positions: tuple[Position, ...] = field(default=())
    cash: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise PortfolioSpecError(
                f"portfolio name must be a non-empty string, got {self.name!r}"
            )
        object.__setattr__(self, "name", self.name.strip())

        # Accept any iterable of positions but store an immutable tuple.
        object.__setattr__(self, "positions", tuple(self.positions))

        cash = _require_number(self.cash, "cash")
        if cash < 0:
            raise PortfolioSpecError(f"cash must be >= 0, got {cash}")
        object.__setattr__(self, "cash", cash)

        # Duplicate detection runs on the NORMALIZED symbols (Position already
        # uppercased them), which is what catches brk-b vs BRK-B.
        seen: set[str] = set()
        for pos in self.positions:
            if pos.symbol in seen:
                raise PortfolioSpecError(f"duplicate symbol {pos.symbol!r}")
            seen.add(pos.symbol)

        # Generator inside sum(): only weight-positions contribute; shares
        # positions have weight=None and are skipped by the `if` clause.
        explicit = sum(p.weight for p in self.positions if p.weight is not None)
        if explicit > 1.0 + _TOL:
            raise PortfolioSpecError(f"explicit weights sum to {explicit:.6f}, must be <= 1")

        if not self.positions and self.cash == 0.0:
            raise PortfolioSpecError("portfolio has no positions and no cash")

    @property
    def symbols(self) -> tuple[str, ...]:
        """Symbols in spec order."""
        return tuple(p.symbol for p in self.positions)


@dataclass(frozen=True, slots=True)
class ResolvedPortfolio:
    """The output of resolve_weights: one weight per symbol, plus cash.

    weights + cash_weight always sum to 1. nav is the dollar NAV implied by
    as-of prices, or None for the scale-free case (only weight positions,
    zero cash), where no dollar amount is knowable.
    """

    name: str
    weights: dict[str, float]  # insertion order == spec order (dicts keep order)
    cash_weight: float
    nav: float | None


def load_portfolio(path: Path | str) -> Portfolio:
    """Parse and validate a portfolio.yaml. See module notes for the rules."""
    path = Path(path)  # accept a plain string path too
    text = path.read_text()  # missing file -> FileNotFoundError, already clear

    try:
        # safe_load, never load: load() can execute arbitrary Python tags
        # embedded in the YAML. There is no reason to ever trust a data file
        # that much.
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        # `from exc` keeps the original parser error chained underneath ours,
        # so the YAML library's line/column info stays visible in tracebacks.
        raise PortfolioSpecError(f"{path}: invalid YAML: {exc}") from exc

    if not isinstance(data, dict):
        # e.g. a file that is just a list, or a bare string.
        raise PortfolioSpecError(f"{path}: top level must be a mapping, got {type(data).__name__}")

    # Set difference finds typos: anything in the file that isn't in the
    # allowed vocabulary. sorted() makes the error message deterministic.
    unknown = set(data) - _ALLOWED_ROOT_KEYS
    if unknown:
        raise PortfolioSpecError(
            f"{path}: unknown key(s) {sorted(unknown)}; allowed: {sorted(_ALLOWED_ROOT_KEYS)}"
        )

    # `positions:` with nothing under it parses as None; treat as empty.
    raw_positions = data.get("positions") or []
    if not isinstance(raw_positions, list):
        raise PortfolioSpecError(f"{path}: `positions` must be a list")

    positions: list[Position] = []
    for i, entry in enumerate(raw_positions):
        # `where` prefixes every error below with file + index, so a bad spec
        # points at its own line: "portfolio.yaml: positions[2]: ..."
        where = f"{path}: positions[{i}]"
        if not isinstance(entry, dict):
            raise PortfolioSpecError(f"{where}: each position must be a mapping, got {entry!r}")
        unknown = set(entry) - _ALLOWED_POSITION_KEYS
        if unknown:
            raise PortfolioSpecError(
                f"{where}: unknown key(s) {sorted(unknown)}; "
                f"allowed: {sorted(_ALLOWED_POSITION_KEYS)}"
            )
        if "symbol" not in entry:
            raise PortfolioSpecError(f"{where}: missing required key `symbol`")
        try:
            # Position.__post_init__ does the field validation; this function
            # only handles YAML structure. .get() returns None for absent
            # keys, which is exactly what Position's optional fields expect.
            positions.append(
                Position(
                    symbol=entry["symbol"],
                    shares=entry.get("shares"),
                    weight=entry.get("weight"),
                )
            )
        except PortfolioSpecError as exc:
            # Re-raise with file and index so the offending line is findable.
            # `from None` suppresses the redundant "during handling of the
            # above exception" traceback -- it's the same error, re-worded.
            raise PortfolioSpecError(f"{where}: {exc}") from None

    try:
        # Portfolio.__post_init__ runs the cross-position rules (duplicates,
        # weight sum, empty-needs-cash) -- same division of labor as above.
        return Portfolio(
            name=data.get("name", ""),  # "" fails name validation with a clear message
            positions=tuple(positions),
            cash=data.get("cash", 0.0),
        )
    except PortfolioSpecError as exc:
        raise PortfolioSpecError(f"{path}: {exc}") from None


def resolve_weights(portfolio: Portfolio, prices: Mapping[str, float]) -> ResolvedPortfolio:
    """Convert a Portfolio into weights at as-of prices.

    `prices` maps symbol -> price and must cover every shares-position;
    weight positions need no price. Returns weights in spec order, summing
    to 1 together with cash_weight. The math is derived in the module notes.
    """
    share_positions = [p for p in portfolio.positions if p.shares is not None]

    # Collect ALL missing prices before raising, so the caller learns the full
    # shopping list in one error instead of one symbol per attempt.
    missing = sorted(p.symbol for p in share_positions if p.symbol not in prices)
    if missing:
        raise PortfolioSpecError(f"no price for symbol(s): {missing}")
    for p in share_positions:
        price = _require_number(prices[p.symbol], f"price for {p.symbol!r}")
        if price <= 0:
            raise PortfolioSpecError(f"price for {p.symbol!r} must be > 0, got {price}")

    # The three quantities from the module-notes algebra:
    # D = dollar value of share legs, W = sum of explicit weights, C = cash.
    dollars = sum(p.shares * float(prices[p.symbol]) for p in share_positions if p.shares)
    explicit = sum(p.weight for p in portfolio.positions if p.weight is not None)
    cash = portfolio.cash

    if not share_positions and cash == 0.0:
        # Scale-free: only weight positions, no dollar anchor anywhere, so
        # NAV is undefined and the weights pass through as written.
        weights = {p.symbol: float(p.weight) for p in portfolio.positions if p.weight}
        cash_weight = max(0.0, 1.0 - explicit)  # clamp float dust like -1e-17
        nav = None
    else:
        # Dollar legs exist (D + C > 0), so NAV is pinned by the algebra:
        # N = (D + C) / (1 - W). That denominator is why W must be < 1 here.
        if explicit >= 1.0 - _TOL:
            raise PortfolioSpecError(
                f"explicit weights sum to {explicit:.6f}, leaving no NAV share "
                "for the share positions and cash in this spec"
            )
        nav = (dollars + cash) / (1.0 - explicit)
        # Build weights in spec order (dict preserves insertion order).
        weights = {}
        for p in portfolio.positions:
            if p.shares is not None:
                weights[p.symbol] = p.shares * float(prices[p.symbol]) / nav
            else:
                weights[p.symbol] = float(p.weight)  # type: ignore[arg-type]
        cash_weight = cash / nav

    # Internal invariant, not input validation: if this fires, the algebra
    # above has a bug. Tests pin it; the assert documents it.
    assert math.isclose(sum(weights.values()) + cash_weight, 1.0, abs_tol=1e-9)

    return ResolvedPortfolio(name=portfolio.name, weights=weights, cash_weight=cash_weight, nav=nav)
