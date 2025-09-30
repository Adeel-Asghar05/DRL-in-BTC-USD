Reinforcement Learning on BTC/USD

This project applies Reinforcement Learning (RL) to the BTC/USD trading market, aiming to create an agent that can learn to make profitable trading decisions (Buy, Sell, Hold) based on historical market data and technical indicators such as RSI, SMA_10, and MACD. The RL agent maximizes cumulative rewards by optimizing its trading strategy over time, and is evaluated through backtesting against historical data.

Write Python code to train a Deep Q-Network (DQN) agent for BTC/USD trading using a custom Gymnasium environment.

Dataset:

- Input is a CSV file with ~70,000 rows of hourly BTC/USD data.
- Columns: OpenTime, Open, High, Low, Close, Volume, SMA_10, RSI, MACD, MACD_Signal.

Environment requirements:

1. **State (Observation):** One row of the dataset (all columns except OpenTime).
2. **Actions:** Discrete {0 = Hold, 1 = Buy, 2 = Sell}.
3. **Reward function:**
   - Compare chosen action with the next hour price movement:
     - If Buy and next Close > current Close → reward = +1
     - If Sell and next Close < current Close → reward = +1
     - If Hold and price change is small (|ΔClose| < threshold) → reward = +1
     - Otherwise → reward = -1
   - This makes reward purely based on next-hour correctness.
4. **Done condition:** End of dataset.
5. **Reset:** Restart from the first row.

Training requirements:

- Use Stable-Baselines3 DQN.
- Wrap environment with DummyVecEnv.
- Train the agent and save the model as `btc_dqn_agent.zip`.

Evaluation requirements:

- If `btc_dqn_agent.zip` exists, load it. Otherwise, train.
- During training and evaluation, print statements for:
  - Episode number
  - Current step, chosen action (Buy/Sell/Hold), next price movement, reward
  - Running accuracy (%) of correct actions

Other details:

- Use PyTorch (via Stable-Baselines3 backend).
- Use matplotlib to plot training reward history at the end.
- Add clear comments for each step of the code.

main idea is to detect direction of vtc in next hour same with some threshhold and +ve or -ve
