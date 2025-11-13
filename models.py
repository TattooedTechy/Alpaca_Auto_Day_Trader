# models.py
import os
import joblib
import glob
from config import log


def load_model(symbol: str, model_type: str) -> any:
    """
    Loads the most recent model for a given symbol and model type.

    The training pipeline (`train.py`) is responsible for creating these models.
    This function simply loads the latest one found in the 'models' directory.

    Args:
        symbol (str): The stock symbol (e.g., 'AAPL').
        model_type (str): The model type from config (e.g., 'random_forest').

    Returns:
        A trained model object if found, otherwise None.
    """
    model_dir = "models"
    pattern = os.path.join(model_dir, f"{symbol}_{model_type}.joblib")
    candidates = glob.glob(pattern)

    if candidates:
        latest_path = max(candidates, key=os.path.getmtime)
        fname = os.path.basename(latest_path)
        log.info(f"Loading existing model for {symbol}: {fname}")
        return joblib.load(latest_path)
    
    log.warning(f"No model file found for {symbol} with type {model_type}. "
                f"Please run train.py to generate models.")
    return None
