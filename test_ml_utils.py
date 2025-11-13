import unittest
import pandas as pd
import numpy as np
from ml_utils import get_triple_barrier_labels

class TestTripleBarrierLabels(unittest.TestCase):

    def setUp(self):
        """Set up a sample price series for testing."""
        dates = pd.to_datetime(pd.date_range(start='2023-01-01', periods=20, freq='min'))
        # Prices: 100 -> 103 (TP hit) -> 98 (SL hit) -> 100 (timeout)
        prices = [
            100, 101, 102, 103, 102, 101, 100, 99, 98, 99,
            100, 100, 100, 100, 100, 100, 100, 100, 100, 100
        ]
        self.prices = pd.Series(prices, index=dates)
        self.tp_mult = 1.02  # 2% take profit
        self.sl_mult = 0.99  # 1% stop loss
        self.max_hold_bars = 5

    def test_take_profit(self):
        """Test a scenario where the take-profit barrier is hit first."""
        # Event starts at time 0, price 100. TP is 102.
        t_events = self.prices.index[:1]
        labels_df = get_triple_barrier_labels(
            self.prices, t_events, self.tp_mult, self.sl_mult, self.max_hold_bars
        )
        
        self.assertEqual(labels_df.loc[t_events[0], 'label'], 1)
        # TP price of 102 is hit at index 2
        self.assertEqual(labels_df.loc[t_events[0], 't_end'], self.prices.index[2])

    def test_stop_loss(self):
        """Test a scenario where the stop-loss barrier is hit first."""
        # Event starts at time 6, price 100. SL is 99.
        t_events = self.prices.index[6:7]
        labels_df = get_triple_barrier_labels(
            self.prices, t_events, self.tp_mult, self.sl_mult, self.max_hold_bars
        )

        self.assertEqual(labels_df.loc[t_events[0], 'label'], -1)
        # SL price of 99 is hit at index 7
        self.assertEqual(labels_df.loc[t_events[0], 't_end'], self.prices.index[7])

    def test_timeout(self):
        """Test a scenario where the time barrier is hit first."""
        # Event starts at time 10, price 100. Barriers are 102 and 99.
        # Price stays at 100 for the next 5 bars.
        t_events = self.prices.index[10:11]
        labels_df = get_triple_barrier_labels(
            self.prices, t_events, self.tp_mult, self.sl_mult, self.max_hold_bars
        )

        self.assertEqual(labels_df.loc[t_events[0], 'label'], 0)
        # End time should be 5 bars after the start
        self.assertEqual(labels_df.loc[t_events[0], 't_end'], self.prices.index[15])

    def test_no_future_data(self):
        """Test an event at the end of the series, which should time out immediately."""
        t_events = self.prices.index[-1:]
        labels_df = get_triple_barrier_labels(
            self.prices, t_events, self.tp_mult, self.sl_mult, self.max_hold_bars
        )
        self.assertEqual(labels_df.loc[t_events[0], 'label'], 0)
        self.assertEqual(labels_df.loc[t_events[0], 't_end'], self.prices.index[-1])

if __name__ == '__main__':
    unittest.main()