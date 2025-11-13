#data_handler.py
import os
from tenacity import retry, stop_after_attempt, wait_fixed
from functools import lru_cache
from datetime import datetime, timedelta, timezone
import pandas as pd
import requests
from features import generate_features
from alpaca.common.exceptions import APIError
from logger import log
from config import data_client, MARKET_OPEN_TIME, MARKET_CLOSE_TIME, POSITION_CACHE_TTL, MAX_RETRIES, EXTENDED_HOURS
from alpaca.data.enums import DataFeed
from alpaca.data.timeframe import TimeFrame
from alpaca.data.requests import StockBarsRequest
from cachetools import TTLCache, cached

# Cache raw historical bars for POSITION_CACHE_TTL seconds
_data_cache = TTLCache(maxsize=128, ttl=POSITION_CACHE_TTL.total_seconds())

# === Fallback Data Fetching ===
@cached(cache=_data_cache)
@retry(stop=stop_after_attempt(MAX_RETRIES), wait=wait_fixed(2), reraise=True)
def fetch_raw_data(symbol: str, days: int = 20) -> pd.DataFrame:
    """
    Fetch raw minute-bars for `symbol`, cached for POSITION_CACHE_TTL,
    and retry on network/API errors.
    """
    end   = datetime.now(timezone.utc)
    start = end - timedelta(days=days)

    try:
        resp = data_client.get_stock_bars(
            StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Minute,
                start=start.isoformat(),
                end=end.isoformat(),
                adjustment="raw",
                feed=DataFeed.SIP
            )
        )
    except (requests.exceptions.RequestException, APIError) as net_err:
        log.error(f"[{symbol}] Network/API error fetching bars: {net_err}")
        fallback = fetch_raw_data(symbol)
        if fallback is not None:
            return fallback
        raise
    bars = resp.df
    log.info(f"[fetch_raw_data] raw bars for {symbol}: {bars.shape}, index span {bars.index.min()} → {bars.index.max()}")
    
    df = (
        bars
        .loc[bars.index.get_level_values(0) == symbol]
        .reset_index()
        .set_index("timestamp")
    )
    df.index = pd.to_datetime(df.index, utc=True)

    # 3) Clip to regular session if needed
    if not EXTENDED_HOURS:
        df = df.between_time(MARKET_OPEN_TIME, MARKET_CLOSE_TIME)

    # 4) Final sanity checks
    if df.empty:
        raise ValueError(f"No data returned for {symbol}")
    if "close" not in df.columns or df["close"].isna().all():
        raise ValueError(f"Missing or invalid close prices for {symbol}")

    return df

@retry(stop=stop_after_attempt(MAX_RETRIES), wait=wait_fixed(2), reraise=True)
@lru_cache(maxsize=128)
def get_historical_data(symbol: str, days: int = 20, with_features: bool = True, scale: bool = False) -> pd.DataFrame:
    """
    The canonical function to fetch historical data.
    - Fetches raw data using a cached, retry-enabled function.
    - Optionally generates features and labels.
    """
    raw_df = fetch_raw_data(symbol, days=days)
    if with_features:
        return generate_features(raw_df, scale=scale)
    return raw_df

# === Load Cached Data ===
def load_cached_data(symbol):
    """
    Attempts to load cached data for a given symbol from disk or other storage.

    Parameters:
    symbol (str): The stock symbol to fetch cached data for.

    Returns:
    pandas.DataFrame: Cached data if available, otherwise None.
    """
    try:
        # Simulated cache loading mechanism
        # Here you could read a CSV, database, or any other storage where the data is cached.
        cached_file = f"cache/{symbol}_historical_data.csv"
        if os.path.exists(cached_file):
            return pd.read_csv(cached_file, index_col="timestamp", parse_dates=["timestamp"])
        return None
    except Exception as e:
        log.warning(f"[{symbol}] Failed to load cached data: {e}")
        return None

# === Validate Incoming Live Data ===
def validate_live_data(data):
    """
    Validates the incoming live market data (e.g., checking for NaN or invalid values).
    
    Parameters:
    data (pandas.DataFrame): The live market data to validate.
    
    Returns:
    bool: True if the data is valid, False if invalid.
    """
    if data is None or data.empty:
        log.error(f"Invalid data: Data is empty or None.")
        return False

    if data["close"].isna().any():
        log.error(f"Invalid data: 'close' price contains NaN values.")
        return False
    
    return True

# === Resampling to Handle Missing Data ===
def resample_data(df, freq="1T"):
    """
    Resamples the data to a higher time frame (e.g., from 1 minute to 5 minutes).

    Parameters:
    df (pandas.DataFrame): The data to resample.
    freq (str): The resampling frequency, e.g., "5T" for 5-minute data.

    Returns:
    pandas.DataFrame: The resampled data.
    """
    try:
        return df.resample(freq).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum"
        }).dropna()
    except Exception as e:
        log.error(f"Error resampling data: {e}")
        return df
