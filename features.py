#features.py
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from logger import log
from data_handler import get_historical_data

# Centralized list of feature columns used for model training and prediction
FEATURE_COLUMNS = [
    "sma_5", "sma_10", "volatility", "momentum", "ema_5", "rsi"
]

# === Prepare Data ===
def prepare_training_data(symbol):
    # Now uses the canonical get_historical_data from data_handler
    df = get_historical_data(symbol, with_features=False)
    
    # Check for empty data or missing 'close' column
    if df.empty or 'close' not in df.columns or df['close'].isnull().all():
        log.error(f"No valid historical data for {symbol}. Skipping trade.")
        return pd.DataFrame()
    
    # Generate features. Labels are now generated in the training pipeline.
    df = generate_features(df)
    return df


# === Feature Engineering ===
def generate_features(df, scale=False, scaler_type="standard"):
    # Check if 'close' column exists
    if "close" not in df.columns:
        log.error("'close' column missing in data for feature generation.")
        raise ValueError("Input DataFrame must contain 'close' column.")

    # Feature generation
    df["return"] = df["close"].pct_change()
    df["price_movement"] = df["return"]
    df["sma_5"] = df["close"].rolling(window=5).mean()
    df["sma_10"] = df["close"].rolling(window=10).mean()
    df["volatility"] = df["return"].rolling(window=5).std()
    df["momentum"] = df["close"] - df["close"].shift(5)
    df["ema_5"] = df["close"].ewm(span=5, adjust=False).mean()
    df["rsi"] = compute_rsi(df["close"], window=14)

    df.dropna(inplace=True)  # Drop rows with NaN values after generating features

    # Feature scaling
    if scale:
        scaler = None
        if scaler_type == "standard":
            scaler = StandardScaler()
        elif scaler_type == "minmax":
            scaler = MinMaxScaler()
        elif scaler_type == "robust":
            scaler = RobustScaler()
        else:
            log.error(f"Scaler type {scaler_type} is not recognized.")
            raise ValueError(f"Scaler type {scaler_type} is not recognized.")

        # Scale selected columns
        df[FEATURE_COLUMNS] = scaler.fit_transform(df[FEATURE_COLUMNS])

    return df


# === Technical Indicator Calculations ===
def compute_rsi(series, window=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)

    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi
