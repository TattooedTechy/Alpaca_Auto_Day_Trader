# Algorithmic Trading Bot

This project is a comprehensive, Python-based framework for developing and deploying algorithmic trading strategies. It uses a robust machine learning pipeline to make trading decisions and connects to the Alpaca API for live market data and trade execution.

## Features

- **Professional ML Pipeline**: Implements a sophisticated machine learning workflow, including a Triple-Barrier Method for realistic labeling, time-series cross-validation to prevent data leakage, and permutation importance for feature analysis.
- **Vectorized Backtesting Engine**: A dedicated `backtest.py` script to evaluate strategy performance over historical data, calculating key metrics like Sharpe Ratio, Total Return, and Max Drawdown.
- **Robust Risk Management**: A multi-faceted risk system with dynamic position sizing based on model confidence, volatility-adjusted entry thresholds, and portfolio-level max drawdown protection.
- **Persistent State Management**: The bot saves its state (`trade_state.pkl`), allowing it to be stopped and restarted without losing track of open positions or pending orders.
- **Resilient Data Handling**: Caches historical data to minimize API calls and uses automatic retries (`tenacity`) to handle transient network or API errors gracefully.
- **Formal Unit Testing**: Includes a `tests/` directory with a unit testing suite to ensure the correctness of core logic components.
- **Live Trading & Paper Trading**: Supports both live and paper trading accounts through Alpaca.
- **Dynamic Configuration**: All trading parameters, thresholds, and settings are managed via a `config.yaml` file.
- **Modular Design**: Code is separated into logical components for models, strategies, data handling, and execution.

## How It Works

The bot's machine learning workflow is designed for robustness to avoid common pitfalls in financial modeling.

1.  **Data Fetching & Caching**: Fetches historical bar data from Alpaca and caches it locally to minimize API calls and speed up subsequent runs.

2.  **Feature Engineering**: Calculates a suite of technical indicators (e.g., SMA, RSI, Bollinger Bands, MACD, Momentum) from the raw price data to serve as predictive features.

3.  **Labeling (Triple-Barrier Method)**: Instead of predicting simple price direction (up/down), the bot uses the **Triple-Barrier Method**. For each data point, it labels the outcome based on which of three barriers is hit first:
    -   **Upper Barrier**: A take-profit limit (e.g., +1.5% price increase).
    -   **Lower Barrier**: A stop-loss limit (e.g., -1.0% price decrease).
    -   **Vertical Barrier**: A time limit for the trade to play out (e.g., 10 bars).
    This creates a more realistic classification target: `(Buy, Sell, Hold)`.

4.  **Time-Series Cross-Validation**: To train and evaluate the model without data leakage, the bot uses `TimeSeriesSplit` from `scikit-learn`. This ensures the model is always trained on past data and validated on future data, simulating live trading.

5.  **Model Training & Tuning**:
    -   For each symbol, a model (e.g., `RandomForestClassifier`) is trained on the engineered features and triple-barrier labels.
    -   Hyperparameters are tuned using `GridSearchCV` combined with the time-series split to find the most performant and generalizable model.
    -   Feature importance is analyzed (e.g., using permutation importance) to understand which indicators are most predictive.

6.  **Live Decision Making**:
    -   The bot subscribes to a real-time data stream.
    -   On each new data bar, it calculates features and feeds them into the trained model to get a prediction (`Buy` or `Sell` signal and confidence).
    -   If the model issues a signal with sufficient confidence, and it passes pre-trade risk checks (volatility, etc.), the bot calculates a position size and executes the trade.

7.  **Position & Risk Management**: Once a trade is open, the bot monitors it against the same take-profit and stop-loss levels defined in the triple-barrier method, ensuring disciplined exits.

## Getting Started

### Prerequisites

- Python 3.9+
- An Alpaca account (either paper or live).

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd <your-repo-directory>
```

### 2. Install Dependencies

It's highly recommended to use a virtual environment to keep your project's dependencies isolated.

**a. Create the virtual environment (only needs to be done once):**

```bash
python3 -m venv venv
```

**b. Activate the environment and install packages:**

```bash
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
```

### 3. Configure the Bot

The bot is configured using two files: `.env` for your secrets and `config.yaml` for trading parameters.

**a. Create a `.env` file for your Alpaca API keys:**

Create a file named `.env` in the root of the project and add your Alpaca keys.

```ini
# For Paper Trading
APCA_API_KEY_ID="YOUR_PAPER_KEY_ID"
APCA_API_SECRET_KEY="YOUR_PAPER_SECRET_KEY"

# For Live Trading
APCA_LIVE_API_KEY_ID="YOUR_LIVE_KEY_ID"
APCA_LIVE_API_SECRET_KEY="YOUR_LIVE_SECRET_KEY"
```

**b. Create and customize your `config.yaml`:**

Copy the example configuration file to create your local `config.yaml`. This file is ignored by Git, so your personal settings won't be committed.

```bash
cp config.yaml.example config.yaml
```

Now, open `config.yaml` and adjust the parameters to fit your desired strategy. You can change the list of symbols, risk parameters, confidence thresholds, and more.

### 4. Train Models

Before you can run the live bot, you need to train the machine learning models using the historical data. The `train.py` script handles this entire process for you.

```bash
# Train models for all symbols listed in your config.yaml
python train.py

# Or, train for specific symbols
python train.py --symbols AAPL MSFT
```

This will generate `.joblib` model files in the `models/` directory and feature importance plots in the `reports/` directory.

### 5. Backtest Your Strategy (Optional, but Recommended)

After training a model, you can run it through the backtester to evaluate its historical performance before deploying it live.

```bash
# Example: Backtest the AAPL model over the last year
python backtest.py --symbol AAPL --model models/AAPL_random_forest.joblib --days 365
```

This will output performance metrics to the console and save a performance chart to the `reports/` directory.

### 6. Run the Bot

Once configured, you can start the bot:

```bash
python main.py
```

The bot will start printing log messages to the console and to `trading_bot.log`, detailing its actions.

### 6. Running Tests

The project includes a suite of unit tests to verify the correctness of the core logic. To run the tests, execute the following command from the root directory:

```bash
python -m unittest discover
```

## Disclaimer

This trading bot is provided for educational purposes only. Trading financial instruments involves substantial risk and is not suitable for all investors. The author and contributors are not responsible for any financial losses you may incur. **Use at your own risk.**

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---