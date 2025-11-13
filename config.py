#config.py
import yaml
from datetime import timedelta, time
from collections import defaultdict
from dotenv import load_dotenv
import os
from logger import log
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.trading.client import TradingClient
from alpaca.data.live import StockDataStream
from pydantic import BaseModel, Field, field_validator
from typing import Dict, Literal, List
from alpaca.data.enums import DataFeed

# === Pydantic schema definitions ===
class AlpacaConfig(BaseModel):
    key: str
    secret: str

class ModeConfig(BaseModel):
    type: Literal['backtest', 'live']
    use_paper: bool
    extended_hours: bool
    model_type: Literal['logistic_regression', 'random_forest']
    symbols: List[str]

class ThresholdsConfig(BaseModel):
    confidence_threshold: float
    confidence_buy: float
    confidence_sell: float

class RiskConfig(BaseModel):
    max_drawdown_pct: float
    max_hold_seconds: int
    min_holding_minutes: int
    trailing_stop_pct: float
    max_position_size: float
    slippage_percentage: float

class MarketHours(BaseModel):
    open: time
    close: time

    @field_validator('close')
    def close_after_open(cls, v, info) -> time:
        open_time = info.data.get('open')
        if open_time and v <= open_time:
            raise ValueError('market_hours.close must be after market_hours.open')
        return v

class FundamentalFilter(BaseModel):
    eps_max: float
    eps_min: float
    pe_max: float
    pe_min: float

class RiskAdjustmentSettings(BaseModel):
    RISK_PER_TRADE: float
    STOP_LOSS_PCT: float
    TAKE_PROFIT_PCT: float

class RiskAdjustments(BaseModel):
    volatility_threshold: float
    by_volatility: Dict[str, RiskAdjustmentSettings]

class RetrySettings(BaseModel):
    max_retries: int = Field(alias='max_retries')
    trade_lock_timeout_seconds: int
    concurrency_limit: int

class ConfigSchema(BaseModel):
    alpaca: AlpacaConfig
    mode: ModeConfig
    thresholds: ThresholdsConfig
    risk: RiskConfig
    market_hours: MarketHours
    fundamental_filters: Dict[str, FundamentalFilter]
    risk_adjustments: RiskAdjustments
    retry: RetrySettings
    position_cache_ttl: int

# ── Load .env ───────────────────────────────────────────────────────────────────
load_dotenv()  # will look for a .env in your working directory

# ── Load & validate YAML ───────────────────────────────────────────────────────
base_dir   = os.path.dirname(__file__)
config_path = os.path.join(base_dir, 'config.yaml')
with open(config_path) as f:
    cfg_dict = yaml.safe_load(f)
config = ConfigSchema.model_validate(cfg_dict)

# === Alpaca credentials ===
# === Alpaca credentials ===
API_KEY    = os.getenv('APCA_API_KEY_ID',       config.alpaca.key)
API_SECRET = os.getenv('APCA_API_SECRET_KEY',   config.alpaca.secret)
LIVE_KEY    = os.getenv("APCA_LIVE_API_KEY_ID",      config.alpaca.key)
LIVE_SECRET = os.getenv("APCA_LIVE_API_SECRET_KEY",  config.alpaca.secret)

assert API_KEY,    "APCA_API_KEY_ID not set"
assert API_SECRET, "APCA_API_SECRET_KEY not set"
assert LIVE_KEY,   "APCA_LIVE_API_KEY_ID not set"
assert LIVE_SECRET,"APCA_LIVE_API_SECRET_KEY not set"

# === Mode settings ===
MODE       = config.mode.type
USE_PAPER  = config.mode.use_paper
MODEL_TYPE = config.mode.model_type
EXTENDED_HOURS = config.mode.extended_hours

# === Alpaca client setup ===
BASE_URL       = 'https://paper-api.alpaca.markets/v2' if USE_PAPER else 'https://api.alpaca.markets'
trading_client = TradingClient(API_KEY, API_SECRET, paper=USE_PAPER)
data_client    = StockHistoricalDataClient(LIVE_KEY, LIVE_SECRET)
stream         = StockDataStream(LIVE_KEY, LIVE_SECRET, feed=DataFeed.SIP, raw_data=False)

print("→ Trading API base URL:", trading_client._base_url)
try:
    acct = trading_client.get_account()
    print("→ Account status:", acct.status)
except Exception as e:
    print("→ ACCOUNT ERROR:", e)

from symbol_loader import build_symbols
# build raw list from Yahoo
raw_symbols = config.mode.symbols
all_syms    = build_symbols(raw_symbols)

# now prune to only those Alpaca deems tradable
assets     = trading_client.get_all_assets()
tradable   = {a.symbol for a in assets if a.tradable and a.status=="active"}
SYMBOLS    = [s for s in all_syms if s in tradable]

# === Retry & concurrency ===
TRADE_CONCURRENCY = config.retry.concurrency_limit
MAX_RETRIES        = config.retry.max_retries
TRADE_LOCK_TIMEOUT = config.retry.trade_lock_timeout_seconds

# === Thresholds ===
CONFIDENCE_THRESHOLD = config.thresholds.confidence_threshold
CONFIDENCE_BUY       = config.thresholds.confidence_buy
CONFIDENCE_SELL      = config.thresholds.confidence_sell

# === Risk parameters ===
MAX_DRAWDOWN_PCT    = config.risk.max_drawdown_pct
MAX_HOLD_SECONDS    = config.risk.max_hold_seconds
MIN_HOLDING_PERIOD  = timedelta(minutes=config.risk.min_holding_minutes)
TRAILING_STOP_PCT   = config.risk.trailing_stop_pct
MAX_POSITION_SIZE   = config.risk.max_position_size
SLIPPAGE_PERCENTAGE = config.risk.slippage_percentage

# === Market hours ===
MARKET_OPEN_TIME  = config.market_hours.open
MARKET_CLOSE_TIME = config.market_hours.close

# === Fundamental filters ===
FUNDAMENTAL_FILTERS = {k: v.model_dump() for k, v in config.fundamental_filters.items()}

# === Risk adjustments ===
VOLATILITY_THRESHOLD       = config.risk_adjustments.volatility_threshold
RISK_ADJUSTMENTS_BY_VOL    = {k: v.model_dump() for k, v in config.risk_adjustments.by_volatility.items()}
# === Risk adjustments ===
VOLATILITY_THRESHOLD    = config.risk_adjustments.volatility_threshold
RISK_ADJUSTMENTS_BY_VOL = {
    k: v.model_dump()
    for k, v in config.risk_adjustments.by_volatility.items()
}

# --- pull the “default” bucket back into standalone constants ---
vol_adj = config.risk_adjustments.by_volatility
if "default" in vol_adj:
    default_bucket = vol_adj["default"]
else:
    # pick the first defined bucket
    default_bucket = next(iter(vol_adj.values()))

RISK_PER_TRADE  = default_bucket.RISK_PER_TRADE
STOP_LOSS_PCT   = default_bucket.STOP_LOSS_PCT
TAKE_PROFIT_PCT = default_bucket.TAKE_PROFIT_PCT

# === In-memory trade-lock registry ===
TRADE_LOCKS = defaultdict(lambda: {'selling': False, 'locked_at': None})

# === Cache settings ===
POSITION_CACHE_TTL = timedelta(seconds=config.position_cache_ttl)
