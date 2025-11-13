# main.py
import warnings

# ignore only the pandas FutureWarning coming from yahoo_fin.stock_info
warnings.filterwarnings(
    "ignore",
    message="Passing literal html to 'read_html' is deprecated",
    category=FutureWarning,
    module="yahoo_fin.stock_info"
)
import asyncio
from functools import partial
import pandas as pd
import glob
import joblib
import os
from config import SYMBOLS, MODE, trading_client, stream, MODEL_TYPE, CONFIDENCE_THRESHOLD, BASE_URL, API_KEY, API_SECRET
from features import generate_features
from models import load_model
from strategies import TrendFollowingStrategy # Assuming this is the primary strategy for now
from trade_execution import on_bar
from logger import log
from profit_tracker import initialize_capital


# === Validate Configuration ===
def validate_configuration():
    assert SYMBOLS, "SYMBOLS list is empty"
    assert MODE in ['backtest', 'live'], "Invalid mode specified"
    assert isinstance(MODE, str), "MODE should be a string"
    assert isinstance(SYMBOLS, list), "SYMBOLS should be a list"
    assert all(isinstance(sym, str) for sym in SYMBOLS), "SYMBOLS should be a list of strings"

# === Main Entrypoint ===
if __name__ == "__main__":
    # Unit Tests
    print("🔑 Loaded API_KEY (first 4 chars):", API_KEY[:4], "…")
    print("🔑 Loaded API_SECRET (first 4 chars):", API_SECRET[:4], "…")
    print("🌐 BASE_URL:", BASE_URL)
    print("📈 SYMBOLS:", SYMBOLS)
    print("⚙️ MODE:", MODE)
    print("📊 MODEL_TYPE:", MODEL_TYPE)
    print("🎯 CONFIDENCE_THRESHOLD:", CONFIDENCE_THRESHOLD)
    print("🔗 trading_client base URL:", trading_client._base_url)

    # Validate Configuration
    try:
        validate_configuration()
        log.info("Configuration validated successfully.")
    except AssertionError as e:
        log.error(f"Configuration validation failed: {e}")
        exit(1)


    async def main():
        try:
            # Initialize the profit tracker with our starting equity
            account = trading_client.get_account()
            initialize_capital(float(account.equity))
            log.info(f"Initialized capital at: ${account.equity}")
            
            models = {}
            strategies = {}

            # Load models and create strategies for each symbol
            for symbol in SYMBOLS:
                log.info(f"Loading model for {symbol}...")
                model = load_model(symbol, MODEL_TYPE)
                if model:
                    models[symbol] = model
                    strategies[symbol] = TrendFollowingStrategy(model) # Or your logic to select a strategy
                    log.info(f"✅ Successfully loaded model and strategy for {symbol}")
                else:
                    log.warning(f"Could not load model for {symbol}. It will not be traded.")

            to_subscribe = list(models.keys())
            callback = partial(on_bar, models=models, strategies=strategies)
            log.info(f"Subscribing to bars for symbols: {to_subscribe}")
            log.info(f"✅ loaded models for {len(to_subscribe)} symbols, subscribing only to those…")
            stream.subscribe_bars(callback, *to_subscribe)
            log.info("Starting stream...")
            # 4) drive the stream._run_forever coroutine yourself
            while True:
                try:
                    await stream._run_forever()    # <-- do not call stream.run()
                except Exception as e:
                    print(f"⚠️ Stream died: {e!r}. reconnecting in 30s…")
                    # make sure any lingering ws is closed before retry
                    try:
                        await stream._close_ws()
                    except:
                        pass
                    await asyncio.sleep(5)

        except Exception as e:
            log.error(f"[STREAM ERROR] {e}", exc_info=True)

    asyncio.run(main())