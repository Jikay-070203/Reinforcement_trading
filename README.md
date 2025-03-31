
# 📈 Mastering Market Peaks and Valleys: Advanced Trading with Deep Reinforcement Learning

This project explores algorithmic trading using **Deep Reinforcement Learning (DRL)** to navigate the highs and lows of financial markets. By applying advanced RL techniques, agents learn to optimize trading strategies on historical market data.

---

## 🚀 Key Features

- 🧠 **Deep RL Algorithms:** Includes implementations of `DQN`, `A2C`, and `PPO` using Stable Baselines3.
- 🏦 **Custom Trading Environment:** Gym-style environment simulating realistic trading scenarios with price history, portfolio balance, and dynamic rewards.
- 📉 **Market Data Handling:** Historical stock and crypto data (e.g., META, BTCUSD) pulled via `tvDatafeed` and processed for modeling.
- 📊 **Performance Metrics:** Evaluation includes PnL, ROI, and Maximum Drawdown.
- 📂 **Notebook Experiments:** Try out and compare different RL models in `qlearning.ipynb`, `a2c.ipynb`, `ppo.ipynb`, and `deep_qlearning.ipynb`.

---

## 🧠 How it Works

1. **Data Collection:** `get_data.py` downloads and cleans OHLC data.
2. **Environment Setup:** `trading_environment.py` provides a trading simulator with 3 discrete actions: Buy, Hold, Sell.
3. **Agent Training:** `functional.py` wraps DRL agents, registers environments, and manages training/evaluation.
4. **Hyperparameters:** `parameters.json` controls symbol, window size, and train/test splits.
5. **Evaluation:** Agents are tested on unseen data and results visualized with trades overlaid on charts.

---

## 🛠 Installation

```bash
pip install -r requirements.txt
# includes: stable-baselines3, gymnasium, tensorflow, pandas, matplotlib, tvDatafeed
```

---

## 📁 Folder Structure

```
.
├── functional.py           # Agent wrapper and evaluation logic
├── trading_environment.py  # Custom Gym environment
├── get_data.py             # Historical market data downloader
├── parameters.json         # Config file for model training/testing
├── *.ipynb                 # Training/testing experiments
├── data/                   # Contains downloaded price data
```

---

## 📈 Actions & Rewards (Simplified)

| Action | Description     | Reward Logic (v2)                        |
|--------|------------------|-----------------------------------------|
| 0      | Buy              | reward = sell_price - current_price     |
| 1      | Hold             | reward based on position (up/down)      |
| 2      | Sell             | reward = current_price - buy_price      |

---

## 📊 Metrics for Model Evaluation

- **PnL (Profit and Loss)**
- **ROI (Return on Investment)**
- **Max Drawdown**

All tracked during backtesting on test set.

---

## 📄 License

Open source for educational and research use in algorithmic trading.
