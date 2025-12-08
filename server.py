from fastapi import FastAPI, HTTPException, Request

from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import graph  # Importing the provided logic

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



class AnalysisRequest(BaseModel):
    symbol: str
    validation_split: int = 20
    lstm_epochs: int = 50
    lstm_batch_size: int = 32
    lstm_window_size: int = 60
    start_date: str = ""
    end_date: str = ""

@app.get("/")
def read_root():
    return {"message": "Market Insight API is running. Please use the frontend to interact."}

@app.post("/analyze")
async def analyze_stock(request: AnalysisRequest):
    try:
        # 1. Configuration
        symbol = request.symbol.upper()
        # Dates: Use provided or defaults
        # Default start: 2000-01-01, Default end: Tomorrow (to include today's data fully)
        start_date = request.start_date if request.start_date else "2015-01-01"
        end_date = request.end_date if request.end_date else (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        
        print(f"DEBUG: Analyzing {symbol} from {start_date} to {end_date}")

        # 2. Download Data
        try:
            df = graph.download_stock_data(symbol, start_date, end_date)
            print(f"DEBUG: Downloaded {len(df)} rows. Last Date: {df['Date'].iloc[-1]}")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to download data for {symbol}: {str(e)}")

        if len(df) < 200:
             raise HTTPException(status_code=400, detail=f"Not enough data found for {symbol}. Try a major stock like AAPL.")

        # 3. Feature Engineering
        try:
            # Keep the last row for charting (latest price), but we'll need to drop it for training
            df_processed_full = graph.engineer_features(df, drop_undefined_target=False)
            print(f"DEBUG: Processing {symbol}. Full Data Range: {df_processed_full['Date'].iloc[0]} to {df_processed_full['Date'].iloc[-1]}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Feature engineering failed: {str(e)}")

        # 4. Prepare Data for Training
        # We need to handle the case where feature engineering drops rows (NaNs)
        df_processed_full = df_processed_full.dropna(subset=[c for c in df_processed_full.columns if c != 'target'])
        
        # For training, we MUST drop the last row where target is unknown
        df_processed = df_processed_full.iloc[:-1].copy()

        
        # Define features and target (matching graph.py usually)
        # We'll try to use graph.py's prepare_data_splits logic if possible, or manual to ensure control
        # Let's see available columns
        potential_features = [
             'Open', 'High', 'Low', 'Close', 'Volume',
            'return_1', 'return_5', 'return_10',
            'RSI', 'MA_10', 'MA_20', 'MACD',
            'momentum_10', 'volatility_10'
        ]
        available_features = [f for f in potential_features if f in df_processed.columns]
        
        X = df_processed[available_features]
        Y = df_processed['target']

        # 5. Train/Test Split
        validation_size = request.validation_split / 100.0
        split_idx = int(len(df_processed) * (1 - validation_size))
        
        X_train, X_valid = X.iloc[:split_idx], X.iloc[split_idx:]
        Y_train, Y_valid = Y.iloc[:split_idx], Y.iloc[split_idx:]

        # 6. Train Models
        # Classical Models
        classical_templates = graph.create_classical_models(Y_train)
        trained_models, results_table = graph.train_classical_models(
            classical_templates, X_train, Y_train, X_valid, Y_valid, available_features
        )
        
        # LSTM Model
        if graph.TENSORFLOW_AVAILABLE:
            lstm_result, lstm_metrics = graph.train_lstm_model(
                df_processed, X_train, Y_train, X_valid, Y_valid, 
                sequence_length=request.lstm_window_size,
                lstm_epochs=request.lstm_epochs,
                lstm_batch_size=request.lstm_batch_size,
                lstm_feature_cols=available_features
            )
            if lstm_result:
                trained_models.append(lstm_result)
                results_table.append(lstm_metrics)

        if not trained_models:
             raise HTTPException(status_code=500, detail="No models trained successfully.")

        # 7. Select Best Model
        # Helper to safely parse metrics
        def get_f1(m):
             try:
                 return float(m['metrics'].get('Validation F1', 0))
             except:
                 return 0
        best_model = max(trained_models, key=get_f1)
        
        # 8. Predict Next Day
        # Use FULL data to predict for tomorrow (using today's close)
        prediction_result = graph.predict_next_day(best_model, df_processed_full, available_features)

        # 9. Prepare Response Data
        # Historical Data for Charting (Full History including today)
        chart_data_df = df_processed_full.copy()
        chart_data_df['Date'] = chart_data_df['Date'].astype(str) if 'Date' in chart_data_df.columns else chart_data_df.index.astype(str)
        chart_data = chart_data_df.to_dict(orient='records')
        
        # Calculate Correlation Matrix for Heatmap
        # Select numeric features only
        corr_matrix = df_processed[available_features + ['target']].corr()
        corr_data = {
            "x": corr_matrix.columns.tolist(),
            "y": corr_matrix.index.tolist(),
            "z": corr_matrix.values.tolist()
        }

        # Serialize detailed model results
        detailed_results = []
        for m in trained_models:
            # Generate extra details: Classification Report, Confusion Matrix, Pred vs Actual
            extra_details = {
                "classification_report": None,
                "confusion_matrix": None,
                "pred_vs_actual": None
            }
            
            try:
                valid_probs = None
                if m['type'] == 'classical':
                    valid_probs = m['model'].predict_proba(X_valid)[:, 1]
                elif m['type'] == 'lstm':
                     # LSTM prediction requires sequences. 
                     # For simplicity, we skip re-prediction for LSTM if complex, but let's try to grab what we can
                     # If graph.py returns it in m, great. If not, we might be missing it for LSTM.
                     # graph.py does calculate valid_probs inside train_lstm_model, but DOES NOT return it in the result dict.
                     # We will rely on classical models having this data for now.
                     valid_probs = None
                
                if valid_probs is not None:
                    # Use standard 0.5 threshold for visualization to show actual separation power
                    # The optimized threshold often forces "All 1s" for minor F1 gains, which looks broken.
                    thresh = 0.5 
                    preds = (valid_probs >= thresh).astype(int)
                    
                    # 1. Class Report
                    from sklearn.metrics import classification_report, confusion_matrix
                    extra_details['classification_report'] = classification_report(Y_valid, preds, output_dict=True)
                    
                    # 2. Confusion Matrix [TP, FP, FN, TN]
                    cm = confusion_matrix(Y_valid, preds)
                    # Convert to standard nested list
                    extra_details['confusion_matrix'] = cm.tolist()

                    # 3. Pred vs Actual (Last 100 points for visualization)
                    # Get correct dates from the original dataframe slice
                    # Y_valid.index corresponds to the indices in df_processed
                    valid_indices = Y_valid.index
                    valid_dates = df_processed.loc[valid_indices, 'Date'].astype(str).tolist()

                    extra_details['pred_vs_actual'] = {
                        "dates": valid_dates[-100:],
                        "actual": Y_valid.tail(100).tolist(),
                        "predicted": preds[-100:].tolist(),
                        "probability": valid_probs[-100:].tolist()
                    }

            except Exception as e:
                print(f"Error generating details for {m['name']}: {e}")
                pass

            model_entry = {
                "name": m["name"],
                "type": m["type"],
                "metrics": m["metrics"],
                "threshold": m["threshold"],
                "walk_forward": m.get("walk_forward"),
                "feature_importances": m.get("feature_importances"),
                "classification_report": m.get("classification_report") or extra_details['classification_report'],
                "confusion_matrix": extra_details['confusion_matrix'],
                "pred_vs_actual": extra_details['pred_vs_actual']
            }
            detailed_results.append(model_entry)

        # EDA: Target Distribution
        target_counts = df_processed['target'].value_counts().to_dict()
        
        # EDA: Feature Distributions (Histograms)
        eda_df = df_processed.tail(500)
        histograms = {}
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in eda_df.columns:
                hist, bin_edges = np.histogram(eda_df[col].dropna(), bins=20)
                histograms[col] = {
                    "counts": hist.tolist(),
                    "bin_edges": bin_edges.tolist()
                }

        return {
            "status": "success",
            "symbol": symbol,
            "results_table": results_table,
            "detailed_results": detailed_results,
            "best_model_name": best_model['name'],
            "best_model_metrics": best_model['metrics'],
            "prediction_result": prediction_result,
            "chart_data": chart_data,
            "eda": {
                "target_counts": target_counts,
                "histograms": histograms,
                "correlation": corr_data
            }
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
