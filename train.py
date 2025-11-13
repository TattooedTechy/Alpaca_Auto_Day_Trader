import argparse
import yaml
import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Project-specific imports
from data_handler import get_historical_data
from features import generate_features, FEATURE_COLUMNS
from ml_utils import (
    get_triple_barrier_labels,
    tune_and_train_model,
    analyze_feature_importance,
)
from config import log, SYMBOLS


def run_training_pipeline(symbols_to_train: list[str], config_path: str = 'config.yaml'):
    """
    Main function to run the full training pipeline for a list of symbols.
    """
    log.info("Loading ML pipeline configuration...")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # We'll need to add an 'ml_pipeline' section to config.yaml later
    ml_params = config.get('ml_pipeline', {})
    barrier_params = ml_params.get('triple_barrier', {
        'tp_mult': 1.015, 'sl_mult': 0.99, 'max_hold_bars': 20
    })
    tuning_params = ml_params.get('hyperparameter_tuning', {
        'param_grid': {
            'n_estimators': [100, 150],
            'max_depth': [10, 15],
            'min_samples_leaf': [10]
        }
    })

    # --- Training loop for each symbol ---
    for symbol in symbols_to_train:
        log.info(f"--- Starting training pipeline for {symbol} ---")

        try:
            # 1. Fetch data and generate features
            raw_df = get_historical_data(symbol, days=90, with_features=False)
            if raw_df.empty:
                log.warning(f"No data for {symbol}, skipping.")
                continue
            
            features_df = generate_features(raw_df.copy())
            log.info(f"Generated {len(features_df.columns)} features for {symbol}.")

            # 2. Generate labels using the Triple-Barrier Method
            events = features_df.index[:-barrier_params['max_hold_bars']]
            labels_df = get_triple_barrier_labels(
                prices=features_df['close'],
                t_events=events,
                **barrier_params
            )
            log.info(f"Generated {len(labels_df)} labels. Distribution:\n{labels_df['label'].value_counts(normalize=True)}")

            # 3. Align features (X) and labels (y)
            aligned_df = features_df.join(labels_df['label'], how='inner').dropna()
            
            # Remove labels where the outcome was timeout (0), as they are not informative for a buy/sell model
            aligned_df = aligned_df[aligned_df['label'] != 0]
            
            if aligned_df.empty:
                log.warning(f"No valid (non-zero) labels for {symbol} after alignment. Skipping.")
                continue

            X = aligned_df[FEATURE_COLUMNS]
            y = aligned_df['label']

            # 4. Split data into training and validation sets (respecting time order)
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
            log.info(f"Data split: {len(X_train)} training samples, {len(X_val)} validation samples.")

            # 5. Tune, train, and save the model
            model_path = f"models/{symbol}_{config['mode']['model_type']}.joblib"
            best_model = tune_and_train_model(
                X=X_train,
                y=y_train,
                param_grid=tuning_params['param_grid'],
                model_path=model_path
            )

            # 6. Analyze and save feature importance plot
            report_path = f"reports/{symbol}_feature_importance.png"
            analyze_feature_importance(
                model=best_model,
                X=X_val,
                y=y_val,
                save_path=report_path
            )

            log.info(f"--- Successfully completed training for {symbol} ---")

        except Exception as e:
            log.error(f"An error occurred during the training pipeline for {symbol}: {e}", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the ML model training pipeline.")
    parser.add_argument(
        '--symbols',
        nargs='+',
        default=SYMBOLS,
        help=f"A list of symbols to train. Defaults to symbols in config.yaml."
    )
    args = parser.parse_args()

    run_training_pipeline(symbols_to_train=args.symbols)