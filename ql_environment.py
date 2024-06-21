import numpy as np

class TradingEnvironment:
    def __init__(self, data):
        self.data = data  # pandas DataFrame with prices and volumes data
        self.state_size = self.data['Category'].nunique()  # Example: price, volume, and three technical indicators
        self.current_step = 0
        self.cash_in_hand = 10000  # Initial balance in dollars
        self.holdings = 0
        self.action_space = [0, 1, 2]  # 0 = hold, 1 = buy, 2 = sell

        self.states_buy = []
        self.states_sell = []
        self.bought_price = 0

    def reset(self):
        self.current_step = 0
        self.cash_in_hand = 10000
        self.initial_cash_in_hand = self.cash_in_hand
        self.holdings = 0
        return self._get_state()

    def step(self, action):
        # Define how environment adapts to the agent's reaction
        # and how profit and loss for the agent is calculated
        current_price = self.get_current_price()
        reward = self.calculate_reward(action, current_price)
        self.current_step += 1
        next_state, done = self._get_state()
        return next_state, reward, done

    def get_current_price(self):
        return self.data.iloc[self.current_step]['Close']
    def calculate_reward(self, action, current_close):
        reward = 0
        if action == 0:  # Buy
            self.holdings = self.cash_in_hand/current_close
            self.states_buy.append(self.data.index[self.current_step])
            self.bought_price = current_close
            self.cash_in_hand = 0
        elif action == 2:  # Sell
            self.cash_in_hand = self.holdings*current_close
            self.states_sell.append(self.data.index[self.current_step])
            reward = max(self.holdings * (current_close - self.bought_price), 0)
            self.holdings = 0
        return reward

    def _get_state(self):
        # Get environment state and agent's balance
        return self.data.iloc[self.current_step], self.current_step==len(self.data)-1

    def calculate_profitability(self):
        if self.cash_in_hand == 0:
            return 100*((self.holdings*self.get_current_price())-self.initial_cash_in_hand)/self.initial_cash_in_hand
        else:
            return 100*(self.cash_in_hand - self.initial_cash_in_hand)/self.initial_cash_in_hand