import os
import random
import numpy as np
import pickle
from collections import deque
import ql_agent
import ql_environment
from ql_data import*
import time

def train_dqn(agent, environment, episodes, replay, batch_size=40):
    data_folder = os.path.join(os.getcwd(), 'Data')
    for e in range(episodes):
        state, done = environment.reset()
        # print(f'Current state is: {state}')
        for time in range(2000):  # Provided each episode has 2000 steps
            action = agent.act(state, environment.cash_in_hand, environment.holdings)
            next_state, reward, done = environment.step(action)
            agent.update_qtable(action, reward, next_state)
            # print(f'Next state is: {next_state}')
            agent.step(state, action, done)
            state = next_state
            if done:
                break
        if e % 100 == 0:
            returns = environment.calculate_profitability()
            print(f'Episode {e} completed. '
                  f'- Steps: {environment.current_step}'
                  f'- Profitability: {returns}')
            if replay:
                agent.replay(batch_size)
                agent.save(os.path.join(data_folder, f"dqn_trading_model_{e}_{returns}.keras"))
            np.save(os.path.join(data_folder,f"q_table{e}_{returns}"), agent.q_table)
            with open(os.path.join(data_folder,f"q_memory{e}_{returns}"), "wb") as fp:  # Pickling
                pickle.dump(agent.memory, fp)

if __name__=='__main__':
    ticker = "BTC"
    new_data = False

    row_data = get_data(ticker, new_data)
    df = row_data.drop(['Open', 'High', 'Low'], axis=1)
    df = aggregate_indicator(df, column='Close', indicator='bollinger', length=10, num_stds=(1.5, 0, -1.5),
                             prefix='Close_BB5_', signal=True)
    df = aggregate_indicator(df, column='Close', indicator='delta_avg', length=10, prefix='delta_avg')
    df['std_10'] = df['Close'].rolling(10).std()
    df = df.iloc[10:]
    df = cat_to_dummies(df, 'Signal', 'Market')
    df = df.drop(['Close_BB5_1.5', 'Close_BB5_0', 'Close_BB5_-1.5'], axis=1)
    cluster_cols = df.columns.to_list()
    cluster_cols.remove('Close')
    test_data = aggregate_categories(df, cluster_cols, n_clusters=25)

    # First step --> initialize TradingEnvironment with data
    env = ql_environment.TradingEnvironment(test_data)
    # Second step --> initialize DQNAgent(state_size, n_categories, action_size, batch_size)
    dqn_agent = ql_agent.DQNAgent(test_data.shape[1], test_data['Category'].nunique(),
                                  action_size=env.action_space, batch_size=40)
    start = time.time()
    train_dqn(dqn_agent, env, 500,True, batch_size=40)
    end = time.time()
    ex_time = end-start
    print(f'Execution time: {ex_time}')



