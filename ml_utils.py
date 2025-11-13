import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, f1_score
from sklearn.inspection import permutation_importance
import joblib
import matplotlib.pyplot as plt
import matplotlib
# Use a non-interactive backend to prevent crashes in environments without a display
matplotlib.use('Agg')
import os


def get_triple_barrier_labels(
    prices: pd.Series,
    t_events: pd.Index,
    tp_mult: float,
    sl_mult: float,
    max_hold_bars: int,
) -> pd.DataFrame:
    """
    Applies the Triple-Barrier Method for labeling financial data.

    Args:
        prices (pd.Series): A pandas Series of asset prices, indexed by timestamp.
        t_events (pd.Index): The timestamps of the events at which we want to
                             generate a label (i.e., the start of a trade).
        tp_mult (float): The multiplier for the take-profit barrier.
                         e.g., 1.015 for a 1.5% profit target.
        sl_mult (float): The multiplier for the stop-loss barrier.
                         e.g., 0.99 for a 1.0% stop-loss.
        max_hold_bars (int): The maximum number of bars to hold the position
                             before exiting (the vertical barrier).

    Returns:
        pd.DataFrame: A DataFrame with labeling information.
    """
    # Create a DataFrame to store the results, indexed by the event times
    out = pd.DataFrame(index=t_events)
    out['label'] = 0
    out['t_end'] = pd.NaT

    # Get the entry prices and calculate the barrier levels
    entry_prices = prices.loc[t_events]
    tp_levels = entry_prices * tp_mult
    sl_levels = entry_prices * sl_mult

    for t_start, sl, tp in zip(t_events, sl_levels, tp_levels):
        # Get the window of prices to search for barrier touches
        start_idx = prices.index.get_loc(t_start)
        # Define the time window for the trade, ensuring it doesn't go past the end of the price series
        end_idx = min(start_idx + max_hold_bars, prices.shape[0] - 1)
        future_prices = prices.iloc[start_idx:end_idx + 1]

        # Find the first time the take-profit or stop-loss is hit
        tp_touch = future_prices[future_prices >= tp].first_valid_index()
        sl_touch = future_prices[future_prices <= sl].first_valid_index()

        # Determine the outcome based on which barrier was touched first
        if tp_touch is not None and (sl_touch is None or tp_touch <= sl_touch):
            out.loc[t_start, 'label'] = 1
            out.loc[t_start, 't_end'] = tp_touch
        elif sl_touch is not None and (tp_touch is None or sl_touch < tp_touch):
            out.loc[t_start, 'label'] = -1
            out.loc[t_start, 't_end'] = sl_touch

    # For events that timed out, set the end time to the vertical barrier
    timeout_events = out[out['label'] == 0].index
    for t_start in timeout_events:
        start_idx = prices.index.get_loc(t_start)
        end_idx = min(start_idx + max_hold_bars, prices.shape[0] - 1)
        out.loc[t_start, 't_end'] = prices.index[end_idx]

    # Add other useful columns for analysis
    out['entry_price'] = entry_prices
    out['tp_price'] = tp_levels
    out['sl_price'] = sl_levels
    return out


def tune_and_train_model(
    X: pd.DataFrame,
    y: pd.Series,
    param_grid: dict,
    n_splits: int = 5,
    model_path: str = None,
) -> RandomForestClassifier:
    """
    Performs hyperparameter tuning using time-series cross-validation and
    trains the final model.
    """
    print("Starting hyperparameter tuning with TimeSeriesSplit...")
    f1_scorer = make_scorer(f1_score, average="weighted")
    tscv = TimeSeriesSplit(n_splits=n_splits)

    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42, class_weight='balanced'),
        param_grid=param_grid,
        scoring=f1_scorer,
        cv=tscv,
        n_jobs=-1,
        verbose=1,
    )

    grid_search.fit(X, y)

    print(f"Best parameters found: {grid_search.best_params_}")
    print(f"Best F1 score (weighted) on validation sets: {grid_search.best_score_:.4f}")

    best_model = grid_search.best_estimator_
    
    print("Final model with best parameters has been trained.")

    if model_path:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(best_model, model_path)
        print(f"Model saved to {model_path}")

    return best_model


def analyze_feature_importance(
    model: RandomForestClassifier,
    X: pd.DataFrame,
    y: pd.Series,
    n_repeats: int = 10,
    save_path: str = None,
):
    """
    Analyzes and plots feature importance using permutation importance.

    Args:
        model: A trained scikit-learn model.
        X (pd.DataFrame): The feature matrix used for validation.
        y (pd.Series): The target labels used for validation.
        n_repeats (int): Number of times to permute a feature.
        save_path (str, optional): Path to save the importance plot.
    """
    print("Calculating feature importance...")
    
    result = permutation_importance(
        model, X, y, n_repeats=n_repeats, random_state=42, n_jobs=-1
    )

    sorted_idx = result.importances_mean.argsort()
    importances = pd.DataFrame(
        result.importances[sorted_idx].T,
        columns=X.columns[sorted_idx],
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    importances.plot.box(vert=False, whis=10, ax=ax)
    ax.set_title("Permutation Importance")
    ax.axvline(x=0, color="k", linestyle="--")
    ax.set_xlabel("Decrease in F1 score")
    fig.tight_layout()
    
    if save_path:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Feature importance plot saved to {save_path}")
    
    plt.show()

    print("\nTop 10 Features by Mean Importance:")
    mean_importances = result.importances_mean[sorted_idx][::-1]
    feature_names = X.columns[sorted_idx][::-1]
    for i, (name, importance) in enumerate(zip(feature_names, mean_importances)):
        if i >= 10: break
        print(f"{i+1}. {name}: {importance:.5f}")