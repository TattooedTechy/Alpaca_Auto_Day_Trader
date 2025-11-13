# trade_state.py
from logger import log
from collections import defaultdict
from datetime import datetime, timedelta, timezone
import pickle
import json
import os

# === TRADE STATE MANAGER ===
class TradeStateManager:
    def __init__(self, state_file=None):
        self.active_positions: defaultdict = defaultdict(lambda: {
            'has_position': False,
            'pending_order': False,
            'size': 0.0,
            'timestamp': None,
            'entry_price': 0.0
        })
        self.state_file = state_file
        self.log = log

        if state_file:
            self.load_state()

    def set_position(self, symbol: str, status: bool, size: float = 0.0):
        now = datetime.now(timezone.utc)

        if status:
            # opening or updating a live position
            self.active_positions[symbol] = {
                'has_position': True,
                'pending_order': False,
                'size': float(size),
                'timestamp': now,
                'entry_price': self.active_positions[symbol].get('entry_price', 0.0) # Preserve entry price
            }
            self.log.info(f"Position for {symbol} set to HELD with size {size}.")
        else:
            # closing or ensuring no position
            self.active_positions[symbol] = {
                'has_position': False,
                'pending_order': False,
                'size': 0.0,
                'timestamp': now,
                'entry_price': 0.0 # Clear entry price on close
            }
            self.log.info(f"Position for {symbol} CLOSED.")

        self.log.debug(f"Updated state for {symbol}: {self.active_positions[symbol]}")
        self.save_state()

    def set_pending_order(self, symbol: str, status: bool):
        now = datetime.now(timezone.utc)
        self.active_positions[symbol]['pending_order'] = status
        self.active_positions[symbol]['timestamp'] = now
        self.log.debug(f"Pending order for {symbol} set to {'ACTIVE' if status else 'INACTIVE'}.")
        self.save_state()

    def adjust_position_size(self, symbol: str, size_delta: float):
        current_size = self.active_positions[symbol]['size']
        new_size = max(0, current_size + size_delta)
        self.set_position(symbol, status=new_size > 0, size=new_size)
        self.log.debug(f"Adjusted position size for {symbol}: {new_size}.")

    def can_trade(self, symbol: str) -> bool:
        state = self.active_positions.get(symbol, {})
        can_trade = not state.get('pending_order', False)
        self.log.debug(f"Can trade {symbol}: {'YES' if can_trade else 'NO'}")
        return can_trade

    def has_position(self, symbol: str) -> bool:
        state = self.active_positions.get(symbol, {})
        return state.get('has_position', False) and state.get('size', 0) > 1e-9 # Use a small epsilon for float comparison

    def get_position(self, symbol: str):
        return self.active_positions.get(symbol, None)

    def save_state(self):
        """
        Atomically save active_positions to the pickle file.
        """
        if self.state_file:
            pickle_path = self.state_file
            temp_path = pickle_path + ".tmp"
            try:
                with open(temp_path, "wb") as f:
                    pickle.dump(dict(self.active_positions), f)
                os.replace(temp_path, pickle_path)
                self.log.info(f"Trade state saved to {pickle_path}.")
            except Exception as e:
                self.log.error(f"Failed to save trade state to {pickle_path}: {e}")
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except Exception:
                        pass

    def load_state(self):
        if self.state_file and os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'rb') as f:
                    loaded_dict = pickle.load(f)
                    # Restore defaultdict behavior after loading from pickle
                    self.active_positions.update(loaded_dict)
                    self.log.info(f"Trade state loaded from {self.state_file}.")
            except FileNotFoundError:
                self.log.warning(f"Trade state file {self.state_file} not found. Starting fresh.")
            except Exception as e:
                self.log.error(f"Failed to load trade state from {self.state_file}: {e}")

    def export_state(self):
        """
        Atomically export active_positions to JSON with ISO8601 timestamps.
        """
        if self.state_file:
            json_path = self.state_file.replace(".pkl", ".json")
            temp_path = json_path + ".tmp"
            try:
                # Build a serializable dict, converting datetimes to ISO strings
                data = {}
                for sym, state in self.active_positions.items():
                    ts = state.get("timestamp")
                    ts_str = ts.isoformat() if isinstance(ts, datetime) else ts
                    data[sym] = {
                        "has_position": state.get("has_position"),
                        "pending_order": state.get("pending_order"),
                        "size": state.get("size"),
                        "timestamp": ts_str,
                        "entry_price": state.get("entry_price"),
                    }
                with open(temp_path, "w") as f:
                    json.dump(data, f, indent=4)
                os.replace(temp_path, json_path)
                self.log.info(f"Trade state exported to {json_path}.")
            except Exception as e:
                self.log.error(f"Failed to export trade state to JSON: {e}")
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except Exception:
                        pass


# === Global shared instance ===
trade_state = TradeStateManager(state_file="trade_state.pkl")
