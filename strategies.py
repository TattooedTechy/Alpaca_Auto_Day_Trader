# strategies.py
from logger import log
from config import FUNDAMENTAL_FILTERS, CONFIDENCE_BUY, CONFIDENCE_SELL, CONFIDENCE_THRESHOLD, stream
from data_handler import get_historical_data
from features import FEATURE_COLUMNS
from trade_state import trade_state
from typing import Callable, Tuple
import pandas as pd
import yfinance as yf


# === MARKET VOLATILITY HELPER ===
def get_market_volatility(symbol):
    """
    Returns the latest volatility value for the given symbol.
    """
    try:
        df = get_historical_data(symbol, days=10, with_features=True)
        latest_volatility = df["volatility"].iloc[-1]
        return latest_volatility
    except Exception as e:
        log.warning(f"Failed to compute volatility for {symbol}: {e}")
        return 0.0  # fallback to 0 volatility

# === STRATEGY BASE ===
class TradingStrategy:
    def __init__(self, model, strategy_type="default", buy_threshold=CONFIDENCE_BUY, sell_threshold=CONFIDENCE_SELL):
        """
        Base class for trading strategies.

        Parameters:
        model (sklearn model): The trained model used for making predictions.
        strategy_type (str): Type of strategy, used to fetch appropriate fundamental filters.
        buy_threshold (float): The confidence threshold for making buy decisions.
        sell_threshold (float): The confidence threshold for making sell decisions.
        """
        self.model = model
        self.strategy_type = strategy_type
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.filters = FUNDAMENTAL_FILTERS.get(strategy_type, FUNDAMENTAL_FILTERS["default"])


    def _passes_fundamental_filters(self, symbol: str) -> bool:
        """
        Check P/E and EPS via yfinance instead of scraping tables.
        """
        try:
            ticker = yf.Ticker(symbol)
            info   = ticker.info

            pe_ratio = info.get("trailingPE") or info.get("forwardPE")
            eps      = info.get("trailingEps") or info.get("forwardEps")

            if pe_ratio is None or eps is None:
                raise ValueError(f"Missing PE/EPS in yfinance.info for {symbol}")

            passes = (
                self.filters["pe_min"] <= pe_ratio <= self.filters["pe_max"]
                and self.filters["eps_min"] <= eps      <= self.filters["eps_max"]
            )

            log.info(
                f"{symbol} - P/E: {pe_ratio:.2f}, EPS: {eps:.2f} → Passes Fundamentals: {passes}"
            )
            return passes

        except Exception as e:
            log.warning(f"Could not retrieve fundamentals for {symbol}: {e}")
            return False

    def decide(self, features_df: pd.DataFrame, symbol: str) -> Tuple[str, bool, float]:
        raise NotImplementedError


# === TREND FOLLOWING STRATEGY ===
class TrendFollowingStrategy(TradingStrategy):
    def __init__(self, model, volatility_factor: float = 1.5):
        super().__init__(model, strategy_type="logistic_regression", buy_threshold=CONFIDENCE_BUY)
        self.volatility_factor = volatility_factor

    def adjust_buy_threshold(self, symbol: str) -> float:
        vol = get_market_volatility(symbol)
        return self.buy_threshold * (1 + self.volatility_factor * vol)

    def decide(self, df, symbol):
        thr = self.adjust_buy_threshold(symbol)
        latest = df[FEATURE_COLUMNS].dropna().iloc[-1:]
        prob = self.model.predict_proba(latest)[0][1]
        has_pos = trade_state.has_position(symbol)

        if not has_pos and prob > thr:
            # enforce growth‐filter fundamentals
            fund_ok = self._passes_fundamental_filters(symbol)
            log.info(f"{symbol} | fundamental filter → pass={fund_ok}")
            if not fund_ok:
                return "none", False, prob
            return "buy", True, prob

        # *new* sell logic: if we do have a position and momentum flips down:
        if has_pos and prob < self.sell_threshold:
            return "sell", True, prob

        return "none", False, prob