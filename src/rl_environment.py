import numpy as np


class ForexEnvironment:

    def __init__(self, data):

        self.data = data.reset_index(drop=True)

        self.current_step = 0

        self.done = False

    def reset(self):

        self.current_step = 0

        self.done = False

        return self.get_state()

    def get_state(self):

        row = self.data.iloc[self.current_step]

        return (
            row["RSI_State"],
            row["MACD_State"],
            row["Trend_State"]
        )

    def step(self, action):

        current_price = self.data.iloc[self.current_step]["Close"]

        next_price = self.data.iloc[self.current_step + 1]["Close"]

        reward = 0

        # BUY
        if action == 0:
            reward = next_price - current_price

        # SELL
        elif action == 1:
            reward = current_price - next_price

        # HOLD
        else:
            reward = 0

        self.current_step += 1

        if self.current_step >= len(self.data) - 1:
            self.done = True

        next_state = self.get_state()

        return next_state, reward, self.done