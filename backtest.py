import argparse
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os

from data_handler import get_historical_data
from features import generate_features, FEATURE_COLUMNS
from config import log, CONFIDENCE_BUY, CONFIDENCE_SELL

def run_backtest(symbol: str, model_path: str, start_days_ago: int = 365, initial_capital: float = 100000.0):
    """
    Runs a vectorized backtest for a given symbol and trained model.

    Args:
        symbol (str): The stock symbol to backtest.
        model_path (str): Path to the trained .joblib model file.
        start_days_ago (int): How many days of historical data to use for the backtest.
        initial_capital (float): The starting capital for the simulation.
    """
    log.info(f"--- Starting Backtest for {symbol} ---")
    log.info(f"Model: {model_path}, Initial Capital: ${initial_capital:,.2f}")

    # 1. Load Model and Data
    if not os.path.exists(model_path):
        log.error(f"Model file not found at {model_path}")
        return

    model = joblib.load(model_path)
    data = get_historical_data(symbol, days=start_days_ago, with_features=True)
    
    if data.empty:
        log.error(f"No historical data found for {symbol} for the last {start_days_ago} days.")
        return

    # 2. Generate Predictions
    X = data[FEATURE_COLUMNS]
    # predict_proba gives [[P(sell), P(buy)]]
    probabilities = model.predict_proba(X)
    data['prob_buy'] = probabilities[:, 1]

    # 3. Generate Trading Signals
    data['signal'] = 0
    # Buy signal: prob > buy_threshold
    data.loc[data['prob_buy'] > CONFIDENCE_BUY, 'signal'] = 1
    # Sell signal: prob < (1 - sell_threshold)
    data.loc[data['prob_buy'] < (1 - CONFIDENCE_SELL), 'signal'] = -1

    # 4. Vectorized Portfolio Simulation
    # Create a position series. 1 means long, 0 means no position.
    # We can only change position when the signal is non-zero.
    # .ffill() simulates holding the position until a new signal.
    data['position'] = data['signal'].replace(0, np.nan).ffill().fillna(0)

    # Calculate strategy returns (daily returns * position of the previous day)
    data['strategy_returns'] = data['return'] * data['position'].shift(1)

    # Calculate cumulative returns for strategy and buy-and-hold
    data['cumulative_strategy_returns'] = (1 + data['strategy_returns']).cumprod()
    data['cumulative_market_returns'] = (1 + data['return']).cumprod()

    # 5. Calculate Performance Metrics
    portfolio_value = initial_capital * data['cumulative_strategy_returns']
    final_value = portfolio_value.iloc[-1]
    total_return = (final_value / initial_capital) - 1

    # Sharpe Ratio (assuming daily returns and 252 trading days a year)
    sharpe_ratio = (data['strategy_returns'].mean() / data['strategy_returns'].std()) * np.sqrt(252)

    # Max Drawdown
    rolling_max = portfolio_value.cummax()
    daily_drawdown = (portfolio_value / rolling_max) - 1
    max_drawdown = daily_drawdown.min()

    log.info("--- Backtest Results ---")
    log.info(f"Final Portfolio Value: ${final_value:,.2f}")
    log.info(f"Total Return: {total_return:.2%}")
    log.info(f"Annualized Sharpe Ratio: {sharpe_ratio:.2f}")
    log.info(f"Max Drawdown: {max_drawdown:.2%}")

    # 6. Plot Results
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(15, 7))
    
    portfolio_value.plot(ax=ax, label='Strategy Equity Curve')
    (initial_capital * data['cumulative_market_returns']).plot(ax=ax, label=f'{symbol} Buy & Hold', linestyle='--')
    
    ax.set_title(f'Backtest Performance: {symbol} vs. Buy & Hold')
    ax.set_ylabel('Portfolio Value ($)')
    ax.set_xlabel('Date')
    ax.legend()
    
    plot_path = f"reports/{symbol}_backtest_performance.png"
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path)
    log.info(f"Backtest plot saved to {plot_path}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a vectorized backtest for a trading model.")
    parser.add_argument(
        '--symbol',
        type=str,
        required=True,
        help="The stock symbol to backtest (e.g., 'AAPL')."
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help="Path to the trained model file (e.g., 'models/AAPL_random_forest.joblib')."
    )
    parser.add_argument(
        '--days',
        type=int,
        default=365,
        help="Number of past days to include in the backtest."
    )
    parser.add_argument(
        '--capital',
        type=float,
        default=100000.0,
        help="Initial capital for the backtest simulation."
    )
    args = parser.parse_args()

    run_backtest(
        symbol=args.symbol,
        model_path=args.model,
        start_days_ago=args.days,
        initial_capital=args.capital
    )