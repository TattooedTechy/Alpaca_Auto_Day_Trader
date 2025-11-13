#trade_execution.py
from datetime import datetime, timezone
from risk_management import (
    check_exit, get_dynamic_position_size, 
    calculate_dynamic_risk_levels, retry_on_failure,
    should_buy, should_sell, cap_position_size_by_risk,lock_trade, unlock_trade,
    calculate_max_drawdown
)
from logger import log
from config import (
    trading_client, MAX_DRAWDOWN_PCT,
    STOP_LOSS_PCT, TAKE_PROFIT_PCT,
    MAX_RETRIES, VOLATILITY_THRESHOLD, TRADE_CONCURRENCY,
    CONFIDENCE_SELL, CONFIDENCE_BUY, EXTENDED_HOURS
)
from data_handler import get_historical_data
from trade_state import trade_state
from tenacity import retry, stop_after_attempt, wait_fixed
from alpaca.data.enums import DataFeed
from order_execution import execute_trade
import pandas as pd
import asyncio
import numpy as np
from typing import Tuple
from utils import is_market_open, get_current_price
import signal
from config import stream, SYMBOLS as symbols
import yfinance as yf
from typing import Tuple
from alpaca.trading.enums import OrderStatus
from typing import Optional
from typing import Any, Dict
from alpaca.common.exceptions import APIError
from profit_tracker import record_profit

# === Custom Exception for Data Fetching ===
class DataFetchError(Exception):
    """Custom exception to handle data fetching errors."""
    pass

trade_semaphore = asyncio.Semaphore(1)
entry_semaphore = asyncio.Semaphore(1)
manage_semaphore = asyncio.Semaphore(1)
    
# === Risk Check Task ===
async def check_risk(symbol: str, equity: float) -> bool:
    from trade_state import trade_state

    # 1) If we don’t already hold this symbol, skip the drawdown check
    if not trade_state.has_position(symbol):
        log.info(f"[check_risk] No existing position for {symbol}, skipping max‐drawdown check")
        return False

    # 2) Otherwise compute your historical drawdown
    max_dd = calculate_max_drawdown(symbol)
    log.info(f"[check_risk] {symbol}: calculated max drawdown {max_dd:.2f}% (threshold {MAX_DRAWDOWN_PCT}%)")
    if max_dd > MAX_DRAWDOWN_PCT:
        log.warning(f"{symbol}: Max drawdown ({max_dd:.2f}%) exceeded")
        return True

    return False

# === Soft Stop-Loss Logic ===
async def soft_stop_loss(symbol, position_size, entry_price):
    current_price = await get_current_price(symbol)
    stop_loss_price = entry_price * (1 - STOP_LOSS_PCT)

    if current_price <= stop_loss_price:
        # Gradual exit (e.g., 20% of position size at a time)
        reduce_size = position_size * 0.2
        await execute_trade(symbol, "sell", reduce_size)  # Sell 20% of the position
        log.info(f"Soft Stop Loss triggered for {symbol}. Reduced position by {reduce_size}.")
        return True
    return False

# === Soft Take-Profit Logic ===
async def soft_take_profit(symbol, position_size, entry_price):
    current_price = await get_current_price(symbol)
    take_profit_price = entry_price * (1 + TAKE_PROFIT_PCT)

    if current_price >= take_profit_price:
        # Gradual exit (e.g., 20% of position size at a time)
        reduce_size = position_size * 0.2
        await execute_trade(symbol, "sell", reduce_size)  # Sell 20% of the position
        log.info(f"Soft Take Profit triggered for {symbol}. Reduced position by {reduce_size}.")
        return True
    return False

async def on_bar(bar, models: Dict[str, Any], strategies: Dict[str, Any]):
    symbol   = bar.symbol
    
    if symbol not in models or symbol not in strategies:
        log.debug(f"[on_bar] {symbol}: no model/strategy loaded, skipping")
        return
    
    model    = models[symbol]
    strategy = strategies[symbol]

    log.info(f"[on_bar] handling {symbol} @ {bar.timestamp}")

    # 1) fetch & build features+labels
    try:
        df = get_historical_data(symbol, with_features=True)
    except DataFetchError as e:
        log.warning(f"[on_bar] Skipping {symbol}: {e}")
        return
    if df.empty:
        log.warning(f"[on_bar] No bars for {symbol}, skipping")
        return
    log.info(f"[on_bar] Got {len(df)} rows of feature data for {symbol}")

    if len(df) < 50:
        log.warning(f"[on_bar] Not enough data for {symbol} (only {len(df)} rows), skipping")
        return

    # 3) strategy-driven entry/exit
    action, decision, confidence = strategy.decide(df, symbol)
    log.info(f"[on_bar] Strategy says {action.upper()} for {symbol} (conf={confidence:.2f})")

    # 1) check Alpaca for any existing position
    if trade_state.has_position(symbol):
        pos_state = trade_state.get_position(symbol)
        qty = pos_state.get('size', 0.0)
        entry_price = pos_state.get('entry_price', 0.0) # Note: entry_price is not currently stored

        # a) If there's a SELL signal, exit fully immediately
        if action.lower() == "sell" and decision and confidence >= CONFIDENCE_SELL:
            current_price = await get_current_price(symbol)
            
            # 1) compute and lock away the profit
            pnl = (current_price - entry_price) * qty
            record_profit(pnl)
            log.info(
                f"[on_bar] SELL SIGNAL for {symbol}, exiting {qty} shares "
                f"@ ${current_price:.2f}, PnL=${pnl:.2f}"
            )

            # 2) send the sell
            await execute_trade(symbol, "sell", qty, by_qty=True)
            trade_state.set_position(symbol, False)

    # 2) No open position in Alpaca: handle a new BUY signal
    if action.lower() == "buy" and decision and confidence >= CONFIDENCE_BUY:
        if await pre_trade_checks(symbol, strategy, confidence, df):
            await trade(symbol, confidence, df, strategy)
        return
    
    # If we have a position, check for exits (like take-profit)
    if trade_state.has_position(symbol):
        await check_exit(symbol, df)
        return

    # --- no-op ---
    log.debug(f"[on_bar] {symbol}: skipped (action={action}, conf={confidence:.2f})")
    log.info(f"[on_bar] No action for {symbol}")
    log.debug(f"[on_bar] {symbol}: done (action={action}, conf={confidence:.2f})")

async def pre_trade_checks(symbol: str, strategy, confidence: float, df: pd.DataFrame) -> bool:
    log.info(f"[pre_trade_checks] {symbol}: conf={confidence:.2f}")

    # 1) Fundamentals via yfinance
    try:
        ticker = yf.Ticker(symbol)
        info   = ticker.info
        pe  = info.get("trailingPE") or info.get("forwardPE")
        eps = info.get("trailingEps") or info.get("epsTrailingTwelveMonths")
    except Exception as e:
        log.warning(f"[pre_trade_checks] Could not fetch fundamentals for {symbol}: {e}")
        return False

    f = strategy.filters
    log.info(
        f"[pre_trade_checks] {symbol} fundamentals: "
        f"P/E={pe} (range {f['pe_min']}–{f['pe_max']}), "
        f"EPS={eps} (range {f['eps_min']}–{f['eps_max']})"
    )

    if pe is None or not (f["pe_min"] <= pe <= f["pe_max"]):
        log.info(f"[pre_trade_checks] Skipping {symbol}: P/E {pe} out of range")
        return False
    if eps is None or not (f["eps_min"] <= eps <= f["eps_max"]):
        log.info(f"[pre_trade_checks] Skipping {symbol}: EPS {eps} out of range")
        return False
     
    # 2) Volatility
    vol = df["close"].pct_change().rolling(window=20).std().iloc[-1]
    log.info(f"[pre_trade_checks] {symbol} volatility: {vol:.4f}, threshold: {VOLATILITY_THRESHOLD:.4f}")
    if pd.notna(vol) and vol > VOLATILITY_THRESHOLD:
        log.info(f"[pre_trade_checks] Skipping {symbol}: Volatility {vol:.4f} exceeds threshold")
        return False

    # 3) All checks passed
    return True

# 2) Everything needed to open a position (the “buy” leg):
async def enter_position(symbol: str, confidence: float, df: pd.DataFrame) -> Tuple[float, int]:
    entry_price = df["close"].iloc[-1]

    # how many dollars you'd like to allocate
    dollar_amt = await get_dynamic_position_size(symbol, confidence)

    # compute the share quantity
    max_bp = float(trading_client.get_account().buying_power)
    max_shares = max_bp / entry_price
    share_qty = min(dollar_amt / entry_price, max_shares)

    if share_qty < 0.0001:   # below Alpaca's min fractional increment
        log.warning(f"[enter_position] Not enough buying power for even a tiny fraction (need ≥0.0001 shares, have {share_qty:.6f})")
        return None, 0.0

    log.info(f"[enter_position] Allocating ≈{share_qty:.6f} shares of {symbol} (~${share_qty*entry_price:.2f})")

    try:
        # Use a limit order for extended hours, market order for regular hours
        order_type = 'limit' if not is_market_open() and EXTENDED_HOURS else 'market'
        limit_price = entry_price if order_type == 'limit' else None
        
        order = await execute_trade(
            symbol, "buy", share_qty, by_qty=True, order_type=order_type, limit_price=limit_price
        )
    except APIError as e:
        msg = getattr(e, "body", str(e))
        if "insufficient buying power" in msg or '"code":40310000' in msg:
            log.warning(f"[enter_position] insufficient buying power for {symbol}, skipping order ({msg})")
            return None, 0.0
        else:
            log.error(f"[enter_position] unexpected API error for {symbol}: {msg}")
            return None, 0.0

    # 7) Place the order
    order_id = getattr(order, "id", getattr(order, "order_id", None))
    log.info(f"[enter_position] Order placed: symbol={symbol}, id={order_id}, qty={dollar_amt}")

    # 8) Update state
    trade_state.set_pending_order(symbol, True)
    trade_state.set_position(symbol, True, size=dollar_amt)
    trade_state.active_positions[symbol]["max_price"] = entry_price

    return entry_price, float(getattr(order, 'filled_qty', 0.0))

# 4) Finally your top‐level trade() simply wires these three together:
async def trade(symbol: str, confidence: float, df: pd.DataFrame, strategy):
    async with trade_semaphore:
        # --- 0) If we already have a position, no new entry ---
        if trade_state.has_position(symbol):
            log.info(f"[trade] Already holding {symbol}, skipping new entry.")
            return

        # 1) Pre-flight checks
        if not await pre_trade_checks(symbol, strategy, confidence, df):
            return

        # 2) Place the buy
        entry_price, position_size = await enter_position(symbol, confidence, df)

        # 3) Short-circuit if no shares were bought
        if entry_price is None or position_size <= 0:
            log.warning(f"[trade] No position opened for {symbol}, skipping management.")
            return
            
        try:
            all_orders = trading_client.get_orders()
            open_statuses = {
                OrderStatus.NEW,
                OrderStatus.PARTIALLY_FILLED,
                OrderStatus.PENDING_NEW,
                OrderStatus.PENDING_CANCEL,
                OrderStatus.PENDING_REPLACE,
                OrderStatus.ACCEPTED,
            }
            open_for_symbol = [
                o for o in all_orders
                if o.symbol == symbol and o.status in open_statuses
            ]
            log.info(f"[trade] Open orders for {symbol}: {open_for_symbol}")
        except Exception as e:
            log.warning(f"[trade] Could not fetch open orders: {e}")

        # The main on_bar loop will now handle exit checks on subsequent bars
        # 5) Return immediately so dispatch_bar can continue
        return

# === Manual Close ===
def close_position(api, symbol):
    try:
        order = api.close_position(symbol)
        log.info(f"[CLOSE] Position for {symbol} closed")
        return order
    except Exception as e:
        log.error(f"[CLOSE ERROR] Failed to close position for {symbol}: {e}")
        return None

async def shutdown():
    log.info("Shutting down live mode: cancelling pending tasks…")
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    try:
        trading_client.close()
    except Exception as e:
        log.error(f"Error closing trading client: {e}")

def run_live():
    loop = asyncio.get_event_loop()
    # The trade_semaphore is already initialized at the module level
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(shutdown()))
    for symbol in symbols:
        stream.subscribe(symbol, on_bar)
    try:
        loop.run_forever()
    finally:
        loop.run_until_complete(shutdown())
