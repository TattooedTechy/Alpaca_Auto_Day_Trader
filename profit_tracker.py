# profit_tracker.py

INITIAL_CAPITAL: float = 0.0
REALIZED_PROFIT: float = 0.0

def initialize_capital(starting_equity: float):
    """Call once at startup."""
    global INITIAL_CAPITAL
    INITIAL_CAPITAL = starting_equity

def record_profit(pnl: float):
    """Accumulate P&L as you take profits."""
    global REALIZED_PROFIT
    REALIZED_PROFIT += pnl

def get_initial_capital() -> float:
    return INITIAL_CAPITAL

def get_realized_profit() -> float:
    return REALIZED_PROFIT
