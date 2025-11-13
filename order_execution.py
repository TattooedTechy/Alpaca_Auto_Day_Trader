# order_execution.py
import asyncio
from tenacity import retry, stop_after_attempt, wait_fixed
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from config import trading_client, MAX_RETRIES
from logger import log

@retry(stop=stop_after_attempt(MAX_RETRIES), wait=wait_fixed(2), reraise=True)
async def execute_trade(
    symbol: str,
    side: str,
    amount: float,
    by_qty: bool = False,
    order_type: str = 'market',
    limit_price: float = None
):
    """
    Place an order. Can be a market or limit order.
    - If by_qty=False, places a notional‐based order (dollar amount).
    - If by_qty=True, places a share‐based order (quantity, can be fractional).
    Must use DAY time-in-force for fractional orders.
    """
    side_enum = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
    order_details = {
        "symbol": symbol,
        "time_in_force": TimeInForce.DAY,
        "side": side_enum,
    }

    # Add qty or notional
    if by_qty:
        order_details["qty"] = str(amount)
    else:
        order_details["notional"] = str(round(amount, 2))

    # Build the appropriate request object
    if order_type == 'limit':
        if not limit_price:
            raise ValueError("limit_price must be provided for limit orders")
        order_details["limit_price"] = str(limit_price)
        req = LimitOrderRequest(**order_details)
        log_msg = f"[ORDER] Submitting LIMIT {side.upper()} for {symbol} at ${limit_price}"
    elif order_type == 'market':
        req = MarketOrderRequest(**order_details)
        log_msg = f"[ORDER] Submitting MARKET {side.upper()} for {symbol}"
    else:
        raise ValueError(f"Unsupported order_type: {order_type}")

    loop = asyncio.get_running_loop()
    order = await loop.run_in_executor(
        None,
        trading_client.submit_order,
        req
    )
    log.info(f"{log_msg} | ID: {order.id}")
    return order
