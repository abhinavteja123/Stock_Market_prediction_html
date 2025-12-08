# 📈 Market Insight - AI Stock Prediction Dashboard

**Market Insight** is an institutional-grade stock analysis and prediction platform. It leverages an ensemble of machine learning models to decode market volatility and generate high-confidence trading signals.

## 🚀 Features

*   **Multi-Model Ensemble**: Synthesizes predictions from **LSTM** (Deep Learning), **XGBoost** (Gradient Boosting), and **Logistic Regression**.
*   **Interactive Dashboard**: A premium, "bento-box" style UI built with React & Framer Motion.
*   **Forecast Timeline**: Visualizes "Actual vs Predicted" outcomes with clear **Green (UP)** and **Red (DOWN)** prediction zones.
*   **Deep Analysis**:
    *   **Market Structure**: Interactive candlestick/line charts.
    *   **Correlation Matrix**: Heatmaps showing relationships between technical indicators.
    *   **Feature Importance**: Identifies which factors (RSI, MACD, Volume) are driving the market.
*   **Real-Time Configuration**: Adjust Training Epochs, Validation Splits, and Lookback Windows on the fly.

## 🛠️ Tech Stack

### Frontend
*   **React** (Vite)
*   **Framer Motion** (Animations)
*   **Plotly.js** (Financial Charts)
*   **Lucide React** (Icons)
*   **CSS Modules** (Premium Dark Theme)

### Backend
*   **Python 3.8+**
*   **FastAPI** (High-performance API)
*   **Pandas & NumPy** (Data Processing)
*   **Scikit-Learn** (Classical ML)
*   **TensorFlow/Keras** (LSTM Networks)
*   **YFinance** (Real-time Data)

## 📦 Installation & Setup

### 1. Backend Setup

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Run the API Server
python server.py
```
*The server will start at `http://localhost:8000`*

### 2. Frontend Setup

```bash
# 1. Navigate to frontend directory
cd frontend

# 2. Install Node dependencies
npm install

# 3. Start the Development Server
npm run dev
```
*The app will be available at `http://localhost:5173` (or similar)*

## 🔮 Usage

1.  Open the **Frontend** in your browser.
2.  Enter a **Stock Ticker** (e.g., `AAPL`, `INFY.NS`, `BTC-USD`).
3.  Select a **Date Range** and **Validation Split**.
4.  Click **INITIALIZE SYSTEM**.
5.  View the generated **Trading Signals**, **Backtest Metrics**, and **Feature Analysis**.

---
*Built for the Future of Quant Trading.*
