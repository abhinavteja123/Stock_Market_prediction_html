import importlib
import importlib.util
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn import metrics
import yfinance as yf
from datetime import datetime, timedelta

# TensorFlow/Keras dynamic import
tf_spec = importlib.util.find_spec("tensorflow")
if tf_spec is not None:
    tf = importlib.import_module("tensorflow")
    keras_models = importlib.import_module("tensorflow.keras.models")
    keras_layers = importlib.import_module("tensorflow.keras.layers")
    keras_callbacks = importlib.import_module("tensorflow.keras.callbacks")
    Sequential = getattr(keras_models, "Sequential")
    LSTM = getattr(keras_layers, "LSTM")
    Dense = getattr(keras_layers, "Dense")
    Dropout = getattr(keras_layers, "Dropout")
    EarlyStopping = getattr(keras_callbacks, "EarlyStopping")
    TENSORFLOW_AVAILABLE = True
else:
    tf = None
    Sequential = None
    LSTM = None
    Dense = None
    Dropout = None
    EarlyStopping = None
    TENSORFLOW_AVAILABLE = False


def find_optimal_threshold(y_true, probas, metric_priority="f1"):
    """Determine the decision threshold that maximizes the chosen metric."""
    thresholds = np.linspace(0.1, 0.9, 81)
    best = {
        "threshold": 0.5,
        "f1": 0.0,
        "accuracy": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "metric": 0.0
    }
    for threshold in thresholds:
        preds = (probas >= threshold).astype(int)
        accuracy = metrics.accuracy_score(y_true, preds)
        f1 = metrics.f1_score(y_true, preds, zero_division=0)
        precision = metrics.precision_score(y_true, preds, zero_division=0)
        recall = metrics.recall_score(y_true, preds, zero_division=0)
        metric_value = f1 if metric_priority == "f1" else accuracy
        if metric_value > best["metric"]:
            best = {
                "threshold": float(threshold),
                "f1": float(f1),
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "metric": float(metric_value)
            }
    return best


def compute_walk_forward_metrics(model_pipeline, X, y, max_splits=5, min_samples_per_split=120):
    """Run a walk-forward (time-series) evaluation for tabular models."""
    if len(X) < min_samples_per_split * 2:
        return None

    possible_splits = len(X) // min_samples_per_split
    n_splits = min(max_splits, max(2, possible_splits))
    if n_splits < 2:
        return None

    tscv = TimeSeriesSplit(n_splits=n_splits)
    records = []

    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
        if len(train_idx) < min_samples_per_split or len(val_idx) == 0:
            continue

        cloned_model = clone(model_pipeline)
        cloned_model.fit(X.iloc[train_idx], y.iloc[train_idx])
        probas = cloned_model.predict_proba(X.iloc[val_idx])[:, 1]
        preds = (probas >= 0.5).astype(int)

        records.append({
            "fold": fold_idx + 1,
            "train_samples": int(len(train_idx)),
            "valid_samples": int(len(val_idx)),
            "accuracy": float(metrics.accuracy_score(y.iloc[val_idx], preds)),
            "f1": float(metrics.f1_score(y.iloc[val_idx], preds, zero_division=0)),
            "roc_auc": float(metrics.roc_auc_score(y.iloc[val_idx], probas))
        })

    if not records:
        return None

    df_records = pd.DataFrame(records)
    summary = {
        "folds": records,
        "mean_accuracy": float(df_records["accuracy"].mean()),
        "mean_f1": float(df_records["f1"].mean()),
        "mean_roc_auc": float(df_records["roc_auc"].mean()),
        "n_folds": int(len(records))
    }
    return summary


def extract_feature_importance(estimator, feature_names):
    """Return normalized feature importance scores for supported estimators."""
    if estimator is None:
        return None

    model = estimator
    if isinstance(estimator, Pipeline):
        model = estimator.named_steps.get("model", estimator)

    if isinstance(model, VotingClassifier):
        aggregated = {}
        count = 0
        for sub_estimator in model.estimators_:
            if isinstance(sub_estimator, tuple):
                sub_estimator = sub_estimator[1]
            sub_importance = extract_feature_importance(sub_estimator, feature_names)
            if sub_importance:
                count += 1
                for feat, score in sub_importance.items():
                    aggregated[feat] = aggregated.get(feat, 0.0) + float(score)
        if count == 0:
            return None
        for feat in aggregated:
            aggregated[feat] /= count
        return dict(sorted(aggregated.items(), key=lambda x: x[1], reverse=True))

    if isinstance(model, Pipeline):
        model = model.named_steps.get("model", model)

    if hasattr(model, "feature_importances_"):
        importances = np.array(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        coef = model.coef_
        if coef.ndim > 1:
            coef = coef[0]
        importances = np.abs(coef.astype(float))
    else:
        return None

    if importances.size != len(feature_names):
        return None

    return dict(sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True))


def calculate_rsi(data, window=14):
    """Calculate Relative Strength Index (RSI)."""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def create_lstm_sequences(data_array, labels_array, window):
    """Create sequences for LSTM training."""
    X_seq, y_seq = [], []
    for idx in range(window, len(data_array)):
        X_seq.append(data_array[idx - window:idx])
        y_seq.append(labels_array[idx])
    return np.array(X_seq), np.array(y_seq)


def download_stock_data(symbol, start_date, end_date):
    """Download stock data from Yahoo Finance."""
    # Ensure end_date includes the last day by adding a buffer if needed
    # yfinance end_date is exclusive.
    if isinstance(end_date, str):
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        end_date = (end_dt + timedelta(days=1)).strftime("%Y-%m-%d")

    df = yf.download(symbol, start=start_date, end=end_date)
    
    # Handle MultiIndex columns from yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    df.reset_index(inplace=True)
    
    # Check if we have the required columns
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    return df


def engineer_features(df, drop_undefined_target=True):
    """Add technical indicators and features to the dataframe."""
    df_processed = df.copy()

    # Initial cleaning
    essential_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    df_processed = df_processed.dropna(subset=essential_cols)

    # Returns and momentum (past-only)
    df_processed['return_1'] = df_processed['Close'].pct_change()
    df_processed['return_5'] = df_processed['Close'].pct_change(5)
    df_processed['return_10'] = df_processed['Close'].pct_change(10)
    df_processed['momentum_10'] = df_processed['Close'] / df_processed['Close'].shift(10) - 1

    # Volatility
    df_processed['volatility_10'] = df_processed['return_1'].rolling(window=10).std()

    # Moving averages and MACD (past-only)
    df_processed['MA_10'] = df_processed['Close'].rolling(window=10).mean()
    df_processed['MA_20'] = df_processed['Close'].rolling(window=20).mean()
    ema_12 = df_processed['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df_processed['Close'].ewm(span=26, adjust=False).mean()
    df_processed['MACD'] = ema_12 - ema_26

    # RSI
    df_processed['RSI'] = calculate_rsi(df_processed['Close'])

    df_processed.replace([np.inf, -np.inf], np.nan, inplace=True)

    feature_fill_cols = [
        'return_1', 'return_5', 'return_10',
        'momentum_10', 'volatility_10',
        'MA_10', 'MA_20', 'MACD', 'RSI'
    ]

    # Only forward-fill to avoid using future data, then drop remaining NaNs
    df_processed[feature_fill_cols] = df_processed[feature_fill_cols].fillna(method='ffill')
    df_processed = df_processed.dropna(subset=feature_fill_cols)

    # Create target: 1 if next day's close is higher, 0 otherwise
    # Target for the last row is NaN (unknown), effectively.
    df_processed['target'] = np.where(df_processed['Close'].shift(-1) > df_processed['Close'], 1, 0)
    
    if drop_undefined_target:
        # Drop the last row where target is invalid/unknown for TRAINING
        df_processed = df_processed.iloc[:-1]

    return df_processed


def prepare_data_splits(df, features_cols, validation_split_pct=20):
    """Split data into training and validation sets."""
    features = df[features_cols]
    target = df['target']

    validation_size = max(int(len(df) * (validation_split_pct / 100)), 1)
    train_size = len(df) - validation_size

    if train_size <= 0:
        raise ValueError("Validation split too large for the available data.")

    X_train = features.iloc[:train_size]
    X_valid = features.iloc[train_size:]
    Y_train = target.iloc[:train_size]
    Y_valid = target.iloc[train_size:]

    return X_train, X_valid, Y_train, Y_valid


def create_classical_models(Y_train):
    """Create classical machine learning model pipelines."""
    # Calculate class balance
    class_counts = Y_train.value_counts()
    n_down = class_counts.get(0, 0)
    n_up = class_counts.get(1, 0)
    scale_pos_weight = n_down / n_up if n_up > 0 else 1.0
    
    xgb_params = {
        "n_estimators": 50,
        "max_depth": 3,
        "learning_rate": 0.1,
        "scale_pos_weight": scale_pos_weight,
        "random_state": 2022,
        "use_label_encoder": False,
        "eval_metric": 'logloss'
    }

    classical_templates = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, class_weight='balanced', random_state=2022))
        ]),
        "SVM (Poly)": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(kernel='poly', probability=True, class_weight='balanced', random_state=2022))
        ]),
        "XGBoost": Pipeline([
            ("scaler", StandardScaler()),
            ("model", XGBClassifier(**xgb_params))
        ]),
        "Random Forest": Pipeline([
            ("model", RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                class_weight='balanced',
                random_state=2022
            ))
        ]),
        "Voting Ensemble": Pipeline([
            ("scaler", StandardScaler()),
            ("model", VotingClassifier(
                estimators=[
                    ("lr", LogisticRegression(max_iter=1000, class_weight='balanced', random_state=2022)),
                    ("xgb", XGBClassifier(**xgb_params)),
                    ("rf", RandomForestClassifier(
                        n_estimators=200,
                        max_depth=10,
                        min_samples_leaf=5,
                        class_weight='balanced',
                        random_state=2022
                    ))
                ],
                voting='soft'
            ))
        ])
    }
    
    return classical_templates


def train_classical_models(classical_templates, X_train, Y_train, X_valid, Y_valid, features_cols):
    """Train all classical ML models and return results."""
    trained_models = []
    results = []
    
    for name, pipeline_template in classical_templates.items():
        pipeline_model = clone(pipeline_template)
        pipeline_model.fit(X_train, Y_train)
        
        train_probs = pipeline_model.predict_proba(X_train)[:, 1]
        valid_probs = pipeline_model.predict_proba(X_valid)[:, 1]
        
        train_auc = metrics.roc_auc_score(Y_train, train_probs)
        valid_auc = metrics.roc_auc_score(Y_valid, valid_probs)
        
        threshold_info = find_optimal_threshold(Y_valid, valid_probs, metric_priority="f1")
        valid_preds_threshold = (valid_probs >= threshold_info["threshold"]).astype(int)
        valid_accuracy = metrics.accuracy_score(Y_valid, valid_preds_threshold)
        valid_f1 = metrics.f1_score(Y_valid, valid_preds_threshold, zero_division=0)
        
        walk_forward_summary = compute_walk_forward_metrics(pipeline_template, X_train, Y_train)
        feature_importances = extract_feature_importance(pipeline_model, features_cols)
        
        trained_models.append({
            "name": name,
            "type": "classical",
            "model": pipeline_model,
            "threshold": threshold_info,
            "walk_forward": walk_forward_summary,
            "feature_importances": feature_importances,
            "metrics": {
                "Train AUC": train_auc,
                "Valid AUC": valid_auc,
                "Validation Accuracy": valid_accuracy,
                "Validation F1": valid_f1,
                "Validation Precision": threshold_info["precision"],
                "Validation Recall": threshold_info["recall"]
            }
        })
        
        results.append({
            'Model': name,
            'Train AUC': f"{train_auc:.4f}",
            'Valid AUC': f"{valid_auc:.4f}",
            'Validation Accuracy': f"{valid_accuracy:.4f}",
            'Validation F1': f"{valid_f1:.4f}",
            'Optimal Threshold': f"{threshold_info['threshold']:.2f}"
        })
    
    return trained_models, results


def train_lstm_model(df, X_train, Y_train, X_valid, Y_valid, sequence_length=60, 
                     lstm_epochs=80, lstm_batch_size=32, lstm_feature_cols=None):
    """Train LSTM model if TensorFlow is available."""
    if not TENSORFLOW_AVAILABLE:
        print("TensorFlow not available. Skipping LSTM training.")
        return None, None
    
    if lstm_feature_cols is None:
        lstm_feature_cols = ['Close', 'Volume', 'RSI', 'MA_20', 'volatility_10']
    
    if len(X_train) <= sequence_length or len(X_valid) <= sequence_length:
        print("Not enough data for LSTM training.")
        return None, None
    
    lstm_scaler = MinMaxScaler()
    train_lstm_features = df[lstm_feature_cols].iloc[:len(X_train)]
    valid_lstm_features = df[lstm_feature_cols].iloc[len(X_train):len(X_train)+len(X_valid)]
    
    train_scaled = lstm_scaler.fit_transform(train_lstm_features)
    valid_scaled = lstm_scaler.transform(valid_lstm_features)
    
    X_train_seq, Y_train_seq = create_lstm_sequences(train_scaled, Y_train.values, sequence_length)
    X_valid_seq, Y_valid_seq = create_lstm_sequences(valid_scaled, Y_valid.values, sequence_length)
    
    if len(X_train_seq) == 0 or len(X_valid_seq) == 0:
        print("Not enough sequences for LSTM training.")
        return None, None
    
    lstm_model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(sequence_length, len(lstm_feature_cols))),
        Dropout(0.3),
        LSTM(32),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    lstm_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    history = lstm_model.fit(
        X_train_seq, Y_train_seq,
        validation_data=(X_valid_seq, Y_valid_seq),
        epochs=lstm_epochs,
        batch_size=lstm_batch_size,
        callbacks=[early_stop],
        verbose=0
    )
    
    train_probs_lstm = lstm_model.predict(X_train_seq, verbose=0).ravel()
    valid_probs_lstm = lstm_model.predict(X_valid_seq, verbose=0).ravel()
    threshold_info_lstm = find_optimal_threshold(Y_valid_seq, valid_probs_lstm, metric_priority="f1")
    valid_preds_lstm = (valid_probs_lstm >= threshold_info_lstm["threshold"]).astype(int)
    
    train_auc_lstm = metrics.roc_auc_score(Y_train_seq, train_probs_lstm)
    valid_auc_lstm = metrics.roc_auc_score(Y_valid_seq, valid_probs_lstm)
    valid_accuracy_lstm = metrics.accuracy_score(Y_valid_seq, valid_preds_lstm)
    valid_f1_lstm = metrics.f1_score(Y_valid_seq, valid_preds_lstm, zero_division=0)
    
    lstm_result = {
        "name": f"LSTM (window={sequence_length})",
        "type": "lstm",
        "model": lstm_model,
        "scaler": lstm_scaler,
        "lstm_features": lstm_feature_cols,
        "sequence_length": sequence_length,
        "threshold": threshold_info_lstm,
        "metrics": {
            "Train AUC": train_auc_lstm,
            "Valid AUC": valid_auc_lstm,
            "Validation Accuracy": valid_accuracy_lstm,
            "Validation F1": valid_f1_lstm,
            "Validation Precision": threshold_info_lstm["precision"],
            "Validation Recall": threshold_info_lstm["recall"]
        }
    }
    
    result_row = {
        'Model': f"LSTM (window={sequence_length})",
        'Train AUC': f"{train_auc_lstm:.4f}",
        'Valid AUC': f"{valid_auc_lstm:.4f}",
        'Validation Accuracy': f"{valid_accuracy_lstm:.4f}",
        'Validation F1': f"{valid_f1_lstm:.4f}",
        'Optimal Threshold': f"{threshold_info_lstm['threshold']:.2f}"
    }
    
    return lstm_result, result_row


def predict_next_day(model_entry, df, features_cols, sequence_length=60):
    """Make a prediction for the next trading day."""
    threshold_value = model_entry.get('threshold', {}).get('threshold', 0.5)
    
    if model_entry['type'] == 'classical':
        latest_features = df[features_cols].iloc[-1:].copy()
        prob_prediction = float(model_entry['model'].predict_proba(latest_features)[0][1])
    else:  # LSTM
        if len(df) < sequence_length:
            raise ValueError("Not enough recent data points to generate an LSTM prediction.")
        
        lstm_features = model_entry.get('lstm_features')
        lstm_scaler = model_entry.get('scaler')
        latest_window = df[lstm_features].iloc[-sequence_length:]
        scaled_window = lstm_scaler.transform(latest_window)
        lstm_sequence = scaled_window.reshape(1, scaled_window.shape[0], scaled_window.shape[1])
        prob_prediction = float(model_entry['model'].predict(lstm_sequence, verbose=0).ravel()[0])
    
    prediction = int(prob_prediction >= threshold_value)
    confidence = (prob_prediction if prediction == 1 else 1 - prob_prediction) * 100
    
    return {
        "prediction": prediction,
        "direction": "UP" if prediction == 1 else "DOWN",
        "probability_up": prob_prediction * 100,
        "confidence": confidence,
        "threshold": threshold_value
    }


# Example usage
if __name__ == "__main__":
    # Configuration
    symbol = "AAPL"
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2024, 12, 1)
    validation_split_pct = 20
    
    features_cols = [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'return_1', 'return_5', 'return_10',
        'RSI', 'MA_10', 'MA_20', 'MACD',
        'momentum_10', 'volatility_10'
    ]
    
    # Download and prepare data
    print(f"Downloading {symbol} data...")
    df = download_stock_data(symbol, start_date, end_date)
    print(f"Downloaded {len(df)} rows")
    
    # Engineer features
    print("Engineering features...")
    df_processed = engineer_features(df)
    print(f"After feature engineering: {len(df_processed)} rows")
    
    # Prepare splits
    X_train, X_valid, Y_train, Y_valid = prepare_data_splits(df_processed, features_cols, validation_split_pct)
    print(f"Training set: {X_train.shape}, Validation set: {X_valid.shape}")
    
    # Train classical models
    print("\nTraining classical models...")
    classical_templates = create_classical_models(Y_train)
    trained_models, results = train_classical_models(
        classical_templates, X_train, Y_train, X_valid, Y_valid, features_cols
    )
    
    # Print results
    print("\nModel Performance:")
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    
    # Train LSTM if available
    if TENSORFLOW_AVAILABLE:
        print("\nTraining LSTM model...")
        lstm_result, lstm_row = train_lstm_model(
            df_processed, X_train, Y_train, X_valid, Y_valid,
            sequence_length=60, lstm_epochs=80, lstm_batch_size=32
        )
        if lstm_result:
            trained_models.append(lstm_result)
            print(f"LSTM - Valid F1: {lstm_result['metrics']['Validation F1']:.4f}")
    
    # Find best model
    best_model = max(trained_models, key=lambda m: m['metrics']['Validation F1'])
    print(f"\nBest Model: {best_model['name']}")
    print(f"Validation F1: {best_model['metrics']['Validation F1']:.4f}")
    
    # Make prediction for next day
    print("\nPredicting next trading day...")
    prediction_result = predict_next_day(best_model, df_processed, features_cols)
    print(f"Prediction: {prediction_result['direction']}")
    print(f"Probability UP: {prediction_result['probability_up']:.2f}%")
    print(f"Confidence: {prediction_result['confidence']:.2f}%")
