from datetime import datetime, time, timezone
from config import MARKET_OPEN_TIME, MARKET_CLOSE_TIME, EXTENDED_HOURS

def is_market_open() -> bool:
    """
    Checks if the market is currently open, respecting the EXTENDED_HOURS setting.
    """
    now_utc = datetime.now(timezone.utc)
    now_time = now_utc.time()

    # If extended hours are enabled, we just check if we are within the broader session.
    # Alpaca's market hours for US equities are 4:00 AM to 8:00 PM EST.
    # 4:00 AM EST is 08:00 or 09:00 UTC. 8:00 PM EST is 00:00 or 01:00 UTC the next day.
    # For simplicity, we rely on Alpaca to reject orders outside the session.
    # A more precise check would be needed for a multi-market bot.
    if EXTENDED_HOURS:
        return True

    # If not extended hours, check against the specific market open/close times.
    return MARKET_OPEN_TIME <= now_time < MARKET_CLOSE_TIME

def get_current_price(symbol: str) -> float:
    """
    A placeholder for a function that would get the most recent price for a symbol.
    In a real implementation, this would use the Alpaca API.
    """
    # This is a mock implementation. In your actual code, you'd use:
    # from config import data_client
    # latest_trade = data_client.get_latest_stock_trade({symbol: symbol})[symbol]
    # return latest_trade.price
    return 150.0 # Mock price for demonstration

def apply_slippage(price: float, percentage: float, side: str) -> float:
    """Mock slippage calculation."""
    if side == 'buy':
        return price * (1 + percentage)
    return price * (1 - percentage)

# Other utility functions like calculate_rr_ratio, has_exceeded_max_hold, etc.
# would also go here.