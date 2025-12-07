# Stock Prediction Dashboard

A professional Machine Learning Dashboard for Stock Prediction, featuring advanced visualizations, probability curves, and multiple model comparisons (Logistic Regression, XGBoost, Random Forest, etc.).

## Features
- **Interactive Charts**: Probability curves with confidence zones, Confusion Matrices, and Price Trends.
- **Multiple Models**: Compare Classical ML models and LSTM.
- **Advanced UI**: Glassmorphism design, smooth animations, and responsive layout.
- **Detailed Metrics**: F1-Score, Accuracy, Precision, Recall, and Feature Importance.

## How to Run Locally

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the Server**:
    ```bash
    python server.py
    ```

3.  **Open in Browser**:
    Go to `http://localhost:8000`

## How to Deploy for Free

### Option 1: Render (Recommended)
1.  Push this code to a GitHub repository.
2.  Sign up at [render.com](https://render.com).
3.  Click **New +** -> **Web Service**.
4.  Connect your GitHub repo.
5.  Use these settings:
    -   **Runtime**: Python 3
    -   **Build Command**: `pip install -r requirements.txt`
    -   **Start Command**: `python server.py` (or `uvicorn server:app --host 0.0.0.0 --port $PORT`)
6.  Click **Create Web Service**. It will be live in minutes!

### Option 2: Hugging Face Spaces
1.  Create a new Space at [huggingface.co/spaces](https://huggingface.co/spaces).
2.  Select **Docker** as the SDK (or standard Python if supported for FastAPI).
3.  Upload these files.
4.  It will build and serve automatically.
