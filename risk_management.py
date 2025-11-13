#risk_management.py
import asyncio
import threading
import random
from datetime import datetime, timedelta, timezone
from order_execution import execute_trade
from logger import log
from config import (
    CONFIDENCE_THRESHOLD, CONFIDENCE_BUY, CONFIDENCE_SELL, RISK_PER_TRADE,
    TAKE_PROFIT_PCT, TRAILING_STOP_PCT, MAX_POSITION_SIZE,
    MIN_HOLDING_PERIOD, MAX_DRAWDOWN_PCT, MAX_RETRIES,
    TRADE_LOCKS, TRADE_LOCK_TIMEOUT, RISK_ADJUSTMENTS_BY_VOL,
    trading_client, STOP_LOSS_PCT
)
from data_handler import get_historical_data
from utils import get_current_price
from trade_state import trade_state
import pandas as pd
from typing import Any
from tenacity import retry, stop_after_attempt, wait_fixed
from typing import Tuple
from profit_tracker import record_profit, get_initial_capital, get_realized_profit



# module‐wide lock to guard TRADE_LOCKS access
_LOCK = threading.Lock()
# keep a rolling history of our profit exits
_profit_exit_history: list[datetime] = []

# === Dynamic Sizing ===
async def get_dynamic_position_size(symbol: str, confidence: float) -> float:
    """
    Returns a dollar notional to commit, anchored to INITIAL_CAPITAL minus any
    profits we've already locked away in REALIZED_PROFIT.
    """
    # 1) Compute the pool we are still willing to risk
    allocatable = get_initial_capital() - get_realized_profit()

    # 2) Base risk-per-trade dollars
    base_risk_dollars = allocatable * RISK_PER_TRADE

    # 3) (Optional) scale by confidence
    sized = base_risk_dollars * confidence

    # 4) Never exceed your max position size (e.g. 5% of allocatable)
    max_notional = allocatable * MAX_POSITION_SIZE
    sized = min(sized, max_notional)

    return sized

# === Risk per Trade Calculation ===
def calculate_risk_per_trade(
    equity: float,
    position_shares: float,
    entry_price: float,
    stop_loss_price: float
) -> float:
    """
    If (entry_price - stop_loss_price)*position_shares / equity
    > RISK_PER_TRADE, reduce the shares so risk == RISK_PER_TRADE.
    """
    risk_amount   = (entry_price - stop_loss_price) * position_shares
    risk_pct      = risk_amount / equity
    if risk_pct > RISK_PER_TRADE:
        return (RISK_PER_TRADE * equity) / (entry_price - stop_loss_price)
    return position_shares

# === Calculate dynamic stops/takes as absolute prices ===

async def calculate_dynamic_risk_levels(symbol: str) -> Tuple[float, float]:
    """
    Returns (stop_price, take_profit_price) based on the
    STOP_LOSS_PCT and TAKE_PROFIT_PCT from your risk buckets.
    """
    # 1) get the current price
    price = await get_current_price(symbol)
    # 2) compute your stop-loss and take-profit levels
    stop_price = price * (1.0 - STOP_LOSS_PCT)
    take_price = price * (1.0 + TAKE_PROFIT_PCT)
    return stop_price, take_price

# === Max Drawdown ===
def calculate_max_drawdown(symbol):
    df = get_historical_data(symbol)
    peak = df["close"].cummax()
    drawdown = (peak - df["close"])/peak
    return drawdown.max() * 100

# === Retry Mechanism (uses MAX_RETRIES from config) ===
async def retry_on_failure(func, delay=5):
    for attempt in range(MAX_RETRIES):
        try:
            return await func()
        except Exception as e:
            log.error(f"Attempt {attempt + 1}/{MAX_RETRIES} failed: {e}")
            if attempt < MAX_RETRIES - 1:
                wait_time = random.uniform(1, 3) * delay
                log.info(f"Retrying in {wait_time:.2f} seconds…")
                await asyncio.sleep(wait_time)
            else:
                log.error(f"Max retries reached for {func}, giving up.")
                raise

# === Entry check now records size & has_position ===
async def check_entry(
    symbol: str,
    model: Any,          # your trained model instance
    df: pd.DataFrame,    # the features DataFrame you just generated
):
    # 0) don't double-trade the same symbol
    if not trade_state.can_trade(symbol):
        log.debug(f"{symbol}: Trade locked or pending. Skipping entry.")
        return

    # 1) liquidity and drawdown checks
    if df["volume"].rolling(10).mean().iloc[-1] < 10_000:
        log.info(f"{symbol}: Low volume, skipping.")
        return

    if calculate_max_drawdown(symbol) > MAX_DRAWDOWN_PCT:
        log.warning(f"{symbol}: Max drawdown exceeded, skipping.")
        return

    # 2) ask model for its probability/confidence
    latest = df.iloc[[-1]]  # 1-row DF
    prob: float = model.predict_proba(latest)[0][1]
    log.info(f"{symbol}: model confidence = {prob:.2f}")

    # 3) only enter if above your buy threshold
    if prob < CONFIDENCE_BUY:
        log.info(f"{symbol}: confidence {prob:.2f} < BUY_THR={CONFIDENCE_BUY:.2f}, skipping")
        return

    # 4) compute your *max* dollars & scale by confidence
    acct     = trading_client.get_account()
    equity   = float(acct.equity)
    max_dollars = equity * MAX_POSITION_SIZE
    dollar_amt  = max_dollars * prob
    log.debug(f"Account equity=${equity:.2f}, max per-symbol=${max_dollars:.2f}, "
              f"allocating ${dollar_amt:.2f} (conf={prob:.2f})")

    # 5) turn into whole shares at last close
    entry_price = df["close"].iloc[-1]
    share_qty   = dollar_amt / entry_price
    if share_qty <= 0.001:
        log.warning(f"{symbol}: computed share_qty={share_qty}, skipping entry")
        return

    # 6) Risk levels (stop-loss & take-profit)
    stop_price, profit_price = calculate_dynamic_risk_levels(symbol)
    share_qty = (calculate_risk_per_trade(equity, share_qty, entry_price, stop_price))

    # 7) submit the order
    trade_state.lock_trade(symbol)
    trade_state.set_pending_order(symbol, True)
    try:
        await retry_on_failure(
            lambda: execute_trade(symbol, "buy", share_qty, by_qty=True)
        )
        # 8) mark your new position
        trade_state.set_position(symbol, True, size=share_qty)
        # Store entry price for later PnL calculation
        trade_state.active_positions[symbol]['entry_price'] = entry_price
    finally:
        trade_state.set_pending_order(symbol, False)

    log.info(f"{symbol}: entered {share_qty} shares @ ${entry_price:.2f}, "
             f"SL=${stop_price:.2f}, TP=${profit_price:.2f}")


# === EXIT ===
async def check_exit(
    symbol: str,
    df: pd.DataFrame
) -> bool:
    # 1) still hold & no concurrent exit
    if (
        not trade_state.has_position(symbol) or
        not trade_state.can_trade(symbol)
    ):
        return False

    pos_state = trade_state.get_position(symbol)
    position_size = pos_state.get('size', 0.0)
    entry_price = pos_state.get('entry_price', 0.0) # Relies on this being set at entry

    # 2) fetch current price
    current_price = await get_current_price(symbol)
    if current_price is None:
        log.error(f"{symbol}: failed to fetch current price")
        return False

    now = datetime.now(timezone.utc)
    entry_time = pos_state.get("timestamp")

    # 3) enforce minimum hold period
    if entry_time and (now - entry_time) < MIN_HOLDING_PERIOD:
        log.debug(f"{symbol}: MIN_HOLDING_PERIOD not met")
        return False

    # 4) compute take-profit level
    take_profit_price = entry_price * (1 + TAKE_PROFIT_PCT)

    log.info(
        f"{symbol}: entry=${entry_price:.2f}, "
        f"TP@${take_profit_price:.2f}, current=${current_price:.2f}"
    )

    # 5) only exit when price ≥ TP
    will_profit = current_price >= take_profit_price
    if not will_profit:
        return False

    # 6) cap same-day profit exits at 3 per rolling 5 days
    cutoff = now - timedelta(days=1)
    global _profit_exit_history
    _profit_exit_history = [ts for ts in _profit_exit_history if ts >= cutoff]

    # if it's the same calendar day as entry, also count previous same-day exits
    if entry_time and entry_time.date() == now.date():
        if len(_profit_exit_history) >= 3:
            log.info(f"{symbol}: profit-exit skipped — 3 already in last 5 days")
            return False

    # 7) fire the sell
    log.warning(f"{symbol}: TAKE PROFIT triggered at ${current_price:.2f}")
    trade_state.lock_trade(symbol)
    trade_state.set_pending_order(symbol, True)
    try:
        await retry(
            stop=stop_after_attempt(MAX_RETRIES),
            wait=wait_fixed(2),
            reraise=True
        )(lambda: execute_trade(symbol, "sell", position_size, by_qty=True))()
        pnl = (current_price - entry_price) * position_size
        record_profit(pnl)
        trade_state.set_position(symbol, False)

        # record this profit exit
        _profit_exit_history.append(now)
    finally:
        trade_state.set_pending_order(symbol, False)
        trade_state.unlock_trade(symbol)

    return True

# === Utility Risk Checks ===

def meets_confidence_threshold(confidence: float) -> bool:
    return confidence >= CONFIDENCE_THRESHOLD

def should_buy(confidence: float, df=None, price=None, equity=None) -> bool:
    """
    Decide whether to buy, based on model confidence (and optionally
    the current dataframe, last price, or account equity if you want to
    extend it later).
    """
    # right now we only look at confidence
    return confidence > CONFIDENCE_BUY

def should_sell(confidence: float, df=None, price=None, equity=None) -> bool:
    """
    Decide whether to sell, based on model confidence (and optionally
    the current dataframe, last price, or account equity if you want to
    extend it later).
    """
    return confidence < CONFIDENCE_SELL

def is_drawdown_exceeded(equity: float, peak_equity: float) -> bool:
    """
    Return True if equity drawdown % exceeds MAX_DRAWDOWN_PCT.
    """
    drawdown_pct = (peak_equity - equity) / peak_equity * 100
    return drawdown_pct > MAX_DRAWDOWN_PCT

def cap_position_size_by_risk(symbol: str, raw_size: float, last_price: float) -> float:
    # pull your per-symbol “max risk per trade” bucket (or default)
    bucket = RISK_ADJUSTMENTS_BY_VOL.get(symbol, RISK_ADJUSTMENTS_BY_VOL['default'])
    max_risk_per_trade = bucket['RISK_PER_TRADE']
    # defensive cast
    max_risk_per_trade = float(max_risk_per_trade)

    # say your equity is passed in or fetched inside here
    account = trading_client.get_account()   # sync call
    equity  = float(account.equity)
    max_dollars = equity * max_risk_per_trade
    max_size = max_dollars / last_price

    # now both raw_size and max_size are floats
    return float(min(raw_size, max_size))

def is_trade_locked(symbol: str) -> bool:
    """
    Return True if a sell‐lock is currently held for this symbol.
    """
    with _LOCK:
        lock = TRADE_LOCKS.get(symbol)
        return bool(lock and lock["selling"])

def lock_trade(symbol: str) -> None:
    """
    Acquire a sell‐lock for the given symbol to prevent concurrent exits.
    """
    with _LOCK:
        TRADE_LOCKS[symbol]["selling"] = True
        TRADE_LOCKS[symbol]["locked_at"] = datetime.now(timezone.utc)

def unlock_trade(symbol: str) -> None:
    """
    Release the sell‐lock for the given symbol.
    """
    with _LOCK:
        TRADE_LOCKS[symbol]["selling"] = False

def has_lock_expired(symbol: str) -> bool:
    """
    Return True if the trade‐lock for this symbol has passed its timeout.
    """
    with _LOCK:
        lock = TRADE_LOCKS.get(symbol)
        if not lock or not lock.get("locked_at"):
            return True
        elapsed = (datetime.now(timezone.utc) - lock["locked_at"]).total_seconds()
        return elapsed > TRADE_LOCK_TIMEOUT

def calculate_position_size_by_risk(equity: float, price: float, confidence: float) -> float:
    """
    Calculates position size using risk percentage and confidence.
    """
    if price <= 0:
        raise ValueError("Price must be greater than zero")
    confidence_factor = max((confidence - CONFIDENCE_THRESHOLD) / (1 - CONFIDENCE_THRESHOLD), 0)
    max_position_value = equity * RISK_PER_TRADE * confidence_factor
    return max_position_value / price

def adjust_risk(symbol):
    return RISK_ADJUSTMENTS_BY_VOL["default"].copy()
