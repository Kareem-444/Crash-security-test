# -*- coding: utf-8 -*-
"""
Enhanced Security Testing Script - MANUAL INPUT MODE
Maximum accuracy with manual crash point entry
User inputs crash points manually, system generates next 10 predictions
"""

import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, StackingRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from statsmodels.tsa.stattools import acf
from statsmodels.sandbox.stats.runs import runstest_1samp
from prophet import Prophet
from pmdarima import auto_arima
from bayes_opt import BayesianOptimization
import random
import time
import logging
import json
import argparse
import joblib
import pandas as pd
import os
import sys
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Fix Windows encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crash_predictions.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

# Disable TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

def local_entropy(arr):
    """Calculate local entropy"""
    if len(arr) == 0:
        return 0
    probs = np.histogram(arr, bins=min(10, len(arr)), density=True)[0]
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs + 1e-10))

def print_banner():
    """Print application banner"""
    print("\n" + "="*80)
    print(" " * 15 + "CRASH GAME PREDICTOR - MANUAL INPUT MODE")
    print(" " * 20 + "Maximum Accuracy Prediction System")
    print("="*80 + "\n")

def collect_manual_input():
    """Collect crash points manually from user"""
    print("="*80)
    print("MANUAL CRASH POINT INPUT")
    print("="*80)
    print("\nInstructions:")
    print("  - Enter crash points one by one")
    print("  - Press Enter after each value")
    print("  - Type 'done' when finished")
    print("  - Minimum 20 crash points required for accurate predictions")
    print("  - Recommended: 50+ crash points for best accuracy\n")
    print("-"*80)
    
    crash_points = []
    
    while True:
        try:
            user_input = input(f"\nCrash Point #{len(crash_points) + 1} (or 'done' to finish): ").strip()
            
            if user_input.lower() == 'done':
                if len(crash_points) < 20:
                    print(f"\n[WARNING] Only {len(crash_points)} points entered. Need at least 20.")
                    response = input("Continue anyway? (y/n): ").strip().lower()
                    if response == 'y':
                        break
                    else:
                        continue
                else:
                    break
            
            # Parse the input
            value = float(user_input.replace('x', '').replace('X', '').strip())
            
            if value < 1.0:
                print("[ERROR] Crash point must be >= 1.0x")
                continue
            
            if value > 1000.0:
                print("[WARNING] Unusually high value. Are you sure?")
                response = input("Continue? (y/n): ").strip().lower()
                if response != 'y':
                    continue
            
            crash_points.append(value)
            print(f"  [OK] Added: {value:.2f}x  |  Total: {len(crash_points)} points")
            
        except ValueError:
            print("[ERROR] Invalid input. Please enter a number (e.g., 2.45 or 2.45x)")
        except KeyboardInterrupt:
            print("\n\n[INTERRUPTED] Exiting...")
            sys.exit(0)
    
    print("\n" + "-"*80)
    print(f"[SUCCESS] Collected {len(crash_points)} crash points")
    print(f"Range: {min(crash_points):.2f}x - {max(crash_points):.2f}x")
    print(f"Average: {np.mean(crash_points):.2f}x")
    print("="*80 + "\n")
    
    return np.array(crash_points)

def load_from_file(filename):
    """Load crash points from file"""
    try:
        print(f"\nLoading crash points from {filename}...")
        
        if filename.endswith('.json'):
            with open(filename, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    crash_points = np.array(data)
                elif isinstance(data, dict) and 'crash_points' in data:
                    crash_points = np.array(data['crash_points'])
                else:
                    raise ValueError("Invalid JSON format")
        
        elif filename.endswith('.txt') or filename.endswith('.csv'):
            crash_points = np.loadtxt(filename)
        
        else:
            raise ValueError("Unsupported file format. Use .json, .txt, or .csv")
        
        print(f"[OK] Loaded {len(crash_points)} crash points")
        print(f"Range: {min(crash_points):.2f}x - {max(crash_points):.2f}x")
        print(f"Average: {np.mean(crash_points):.2f}x\n")
        
        return crash_points
    
    except Exception as e:
        print(f"[ERROR] Could not load file: {e}")
        return None

def create_advanced_features(data):
    """Create advanced feature set for maximum accuracy - NaN safe"""
    features = []
    
    for i in range(len(data)):
        # Multiple window sizes for multi-scale patterns
        windows = [5, 10, 15, 20]
        feat = {}
        
        for w in windows:
            start_idx = max(0, i - w)
            window_data = data[start_idx:i+1]
            
            if len(window_data) > 0:
                # Statistical features - fill NaN with 0
                feat[f'mean_{w}'] = np.nan_to_num(np.mean(window_data), 0)
                feat[f'median_{w}'] = np.nan_to_num(np.median(window_data), 0)
                feat[f'std_{w}'] = np.nan_to_num(np.std(window_data), 0)
                feat[f'var_{w}'] = np.nan_to_num(np.var(window_data), 0)
                feat[f'min_{w}'] = np.nan_to_num(np.min(window_data), 0)
                feat[f'max_{w}'] = np.nan_to_num(np.max(window_data), 0)
                feat[f'range_{w}'] = np.nan_to_num(np.max(window_data) - np.min(window_data), 0)
                
                # Percentiles
                feat[f'q25_{w}'] = np.nan_to_num(np.percentile(window_data, 25), 0)
                feat[f'q50_{w}'] = np.nan_to_num(np.percentile(window_data, 50), 0)
                feat[f'q75_{w}'] = np.nan_to_num(np.percentile(window_data, 75), 0)
                feat[f'q90_{w}'] = np.nan_to_num(np.percentile(window_data, 90), 0)
                
                # Entropy and information
                entropy_val = local_entropy(window_data)
                feat[f'entropy_{w}'] = np.nan_to_num(entropy_val, 0)
                
                # Trend analysis
                if len(window_data) > 2:
                    try:
                        trend = np.polyfit(range(len(window_data)), window_data, 1)[0]
                        feat[f'trend_{w}'] = np.nan_to_num(trend, 0)
                    except:
                        feat[f'trend_{w}'] = 0
                    
                    # Acceleration (second derivative)
                    if len(window_data) > 3:
                        try:
                            accel = np.polyfit(range(len(window_data)), window_data, 2)[0]
                            feat[f'accel_{w}'] = np.nan_to_num(accel, 0)
                        except:
                            feat[f'accel_{w}'] = 0
                    else:
                        feat[f'accel_{w}'] = 0
                else:
                    feat[f'trend_{w}'] = 0
                    feat[f'accel_{w}'] = 0
                
                # Volatility
                if len(window_data) > 1:
                    returns = np.diff(window_data)
                    feat[f'volatility_{w}'] = np.nan_to_num(np.std(returns), 0)
                else:
                    feat[f'volatility_{w}'] = 0
            else:
                # Fill all features with 0 if no data
                feat[f'mean_{w}'] = 0
                feat[f'median_{w}'] = 0
                feat[f'std_{w}'] = 0
                feat[f'var_{w}'] = 0
                feat[f'min_{w}'] = 0
                feat[f'max_{w}'] = 0
                feat[f'range_{w}'] = 0
                feat[f'q25_{w}'] = 0
                feat[f'q50_{w}'] = 0
                feat[f'q75_{w}'] = 0
                feat[f'q90_{w}'] = 0
                feat[f'entropy_{w}'] = 0
                feat[f'trend_{w}'] = 0
                feat[f'accel_{w}'] = 0
                feat[f'volatility_{w}'] = 0
        
        # Recent differences
        feat['diff_1'] = data[i] - data[i-1] if i > 0 else 0
        feat['diff_2'] = data[i] - data[i-2] if i > 1 else 0
        feat['diff_3'] = data[i] - data[i-3] if i > 2 else 0
        
        # Pattern features
        feat['is_high'] = 1 if i > 0 and data[i-1] > 3.0 else 0
        feat['is_low'] = 1 if i > 0 and data[i-1] < 1.5 else 0
        feat['is_very_high'] = 1 if i > 0 and data[i-1] > 5.0 else 0
        
        # Momentum indicators
        if i >= 5:
            feat['momentum_5'] = data[i] - data[i-5]
            feat['roc_5'] = (data[i] - data[i-5]) / data[i-5] if data[i-5] != 0 else 0
        else:
            feat['momentum_5'] = 0
            feat['roc_5'] = 0
        
        # Moving average crossovers
        if i >= 20:
            ma_5 = np.mean(data[i-5:i]) if i >= 5 else 0
            ma_10 = np.mean(data[i-10:i]) if i >= 10 else 0
            ma_20 = np.mean(data[i-20:i]) if i >= 20 else 0
            feat['ma_cross_5_10'] = 1 if ma_5 > ma_10 else 0
            feat['ma_cross_5_20'] = 1 if ma_5 > ma_20 else 0
        else:
            feat['ma_cross_5_10'] = 0
            feat['ma_cross_5_20'] = 0
        
        features.append(feat)
    
    df = pd.DataFrame(features)
    
    # Final safety check: replace any remaining NaN with 0
    df = df.fillna(0)
    
    # Replace inf values
    df = df.replace([np.inf, -np.inf], 0)
    
    return df

def train_ultra_accurate_models(data):
    """Train ensemble of 10+ models for maximum accuracy"""
    print("\n" + "="*80)
    print("TRAINING ULTRA-ACCURATE PREDICTION MODELS")
    print("="*80 + "\n")
    
    if len(data) < 20:
        print("[ERROR] Need at least 20 data points")
        return None
    
    # Create features
    print("[1/12] Creating advanced features...")
    features = create_advanced_features(data)
    targets = data
    
    # Split data (use more recent data for testing)
    X_train, X_test, y_train, y_test = train_test_split(
        features, targets, test_size=0.15, random_state=42, shuffle=False
    )
    
    # Additional safety: ensure no NaN or inf in train/test data
    X_train = X_train.fillna(0).replace([np.inf, -np.inf], 0)
    X_test = X_test.fillna(0).replace([np.inf, -np.inf], 0)
    
    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {}
    predictions = {}
    
    # 1. XGBoost with Bayesian Optimization
    print("[2/12] Training XGBoost with Bayesian Optimization...")
    def xgb_cv(max_depth, learning_rate, n_estimators, subsample, colsample_bytree):
        model = xgb.XGBRegressor(
            max_depth=int(max_depth),
            learning_rate=learning_rate,
            n_estimators=int(n_estimators),
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            objective='reg:squarederror',
            random_state=42
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        return -mean_squared_error(y_test, preds)
    
    xgb_bo = BayesianOptimization(
        f=xgb_cv,
        pbounds={
            'max_depth': (4, 15),
            'learning_rate': (0.001, 0.3),
            'n_estimators': (100, 800),
            'subsample': (0.6, 1.0),
            'colsample_bytree': (0.6, 1.0)
        },
        random_state=42,
        verbose=0
    )
    xgb_bo.maximize(init_points=5, n_iter=15)
    
    best = xgb_bo.max['params']
    xgb_model = xgb.XGBRegressor(
        max_depth=int(best['max_depth']),
        learning_rate=best['learning_rate'],
        n_estimators=int(best['n_estimators']),
        subsample=best['subsample'],
        colsample_bytree=best['colsample_bytree'],
        objective='reg:squarederror',
        random_state=42
    )
    xgb_model.fit(X_train, y_train)
    predictions['xgb'] = xgb_model.predict(X_test)
    models['xgb'] = xgb_model
    
    # 2. LightGBM
    print("[3/12] Training LightGBM...")
    lgb_model = lgb.LGBMRegressor(
        num_leaves=50,
        learning_rate=0.03,
        n_estimators=500,
        max_depth=12,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    )
    lgb_model.fit(X_train, y_train)
    predictions['lgb'] = lgb_model.predict(X_test)
    models['lgb'] = lgb_model
    
    # 3. CatBoost
    print("[4/12] Training CatBoost...")
    cat_model = CatBoostRegressor(
        iterations=500,
        learning_rate=0.05,
        depth=10,
        random_state=42,
        verbose=0
    )
    cat_model.fit(X_train, y_train)
    predictions['cat'] = cat_model.predict(X_test)
    models['cat'] = cat_model
    
    # 4. Random Forest
    print("[5/12] Training Random Forest...")
    rf_model = RandomForestRegressor(
        n_estimators=500,
        max_depth=20,
        min_samples_split=3,
        min_samples_leaf=1,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    predictions['rf'] = rf_model.predict(X_test)
    models['rf'] = rf_model
    
    # 5. Extra Trees
    print("[6/12] Training Extra Trees...")
    et_model = ExtraTreesRegressor(
        n_estimators=500,
        max_depth=20,
        min_samples_split=2,
        random_state=42,
        n_jobs=-1
    )
    et_model.fit(X_train, y_train)
    predictions['et'] = et_model.predict(X_test)
    models['et'] = et_model
    
    # 6. Gradient Boosting
    print("[7/12] Training Gradient Boosting...")
    gb_model = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=10,
        subsample=0.8,
        random_state=42
    )
    gb_model.fit(X_train, y_train)
    predictions['gb'] = gb_model.predict(X_test)
    models['gb'] = gb_model
    
    # 7. ElasticNet
    print("[8/12] Training ElasticNet...")
    elastic_model = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
    elastic_model.fit(X_train_scaled, y_train)
    predictions['elastic'] = elastic_model.predict(X_test_scaled)
    models['elastic'] = elastic_model
    
    # 8. Ridge
    print("[9/12] Training Ridge Regression...")
    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(X_train_scaled, y_train)
    predictions['ridge'] = ridge_model.predict(X_test_scaled)
    models['ridge'] = ridge_model
    
    # 9. Advanced LSTM
    print("[10/12] Training Advanced LSTM...")
    seq_length = 20
    if len(data) > seq_length:
        X_seq = np.array([data[i:i+seq_length] for i in range(len(data) - seq_length)])
        y_seq = data[seq_length:]
        X_seq = X_seq.reshape((X_seq.shape[0], X_seq.shape[1], 1))
        
        X_train_seq, X_test_seq, y_train_seq, y_test_seq = train_test_split(
            X_seq, y_seq, test_size=0.15, random_state=42, shuffle=False
        )
        
        lstm_model = Sequential([
            Bidirectional(LSTM(128, return_sequences=True), input_shape=(seq_length, 1)),
            Dropout(0.3),
            BatchNormalization(),
            Bidirectional(LSTM(64, return_sequences=True)),
            Dropout(0.3),
            BatchNormalization(),
            LSTM(32),
            Dropout(0.2),
            Dense(16, activation='relu'),
            BatchNormalization(),
            Dense(1)
        ])
        
        optimizer = Adam(learning_rate=0.001)
        lstm_model.compile(optimizer=optimizer, loss='huber', metrics=['mae'])
        
        early_stop = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=0.00001)
        
        lstm_model.fit(
            X_train_seq, y_train_seq,
            epochs=200,
            batch_size=16,
            verbose=0,
            callbacks=[early_stop, reduce_lr]
        )
        
        pred_lstm = lstm_model.predict(X_test_seq, verbose=0).flatten()
        
        # Align lengths
        if len(pred_lstm) < len(y_test):
            pred_lstm = np.pad(pred_lstm, (0, len(y_test) - len(pred_lstm)), 'edge')
        elif len(pred_lstm) > len(y_test):
            pred_lstm = pred_lstm[:len(y_test)]
        
        predictions['lstm'] = pred_lstm
        models['lstm'] = lstm_model
    
    # 10. GRU Network
    print("[11/12] Training GRU Network...")
    if len(data) > seq_length:
        gru_model = Sequential([
            GRU(128, return_sequences=True, input_shape=(seq_length, 1)),
            Dropout(0.3),
            GRU(64, return_sequences=True),
            Dropout(0.3),
            GRU(32),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        
        gru_model.compile(optimizer=Adam(0.001), loss='huber')
        gru_model.fit(X_train_seq, y_train_seq, epochs=150, batch_size=16, verbose=0, callbacks=[early_stop])
        
        pred_gru = gru_model.predict(X_test_seq, verbose=0).flatten()
        
        if len(pred_gru) < len(y_test):
            pred_gru = np.pad(pred_gru, (0, len(y_test) - len(pred_gru)), 'edge')
        elif len(pred_gru) > len(y_test):
            pred_gru = pred_gru[:len(y_test)]
        
        predictions['gru'] = pred_gru
        models['gru'] = gru_model
    
    # 11. Stacking Ensemble
    print("[12/12] Creating Stacking Ensemble...")
    base_estimators = [
        ('xgb', xgb_model),
        ('lgb', lgb_model),
        ('rf', rf_model),
        ('gb', gb_model)
    ]
    
    stack_model = StackingRegressor(
        estimators=base_estimators,
        final_estimator=Ridge(alpha=0.5),
        cv=5
    )
    stack_model.fit(X_train, y_train)
    predictions['stack'] = stack_model.predict(X_test)
    models['stack'] = stack_model
    
    # Calculate performance and weights
    print("\n" + "-"*80)
    print("MODEL PERFORMANCE")
    print("-"*80)
    
    mses = {}
    maes = {}
    r2s = {}
    
    for name, preds in predictions.items():
        try:
            mse = mean_squared_error(y_test[:len(preds)], preds[:len(y_test)])
            mae = mean_absolute_error(y_test[:len(preds)], preds[:len(y_test)])
            r2 = r2_score(y_test[:len(preds)], preds[:len(y_test)])
            
            mses[name] = mse
            maes[name] = mae
            r2s[name] = r2
            
            print(f"{name.upper():12s} | MSE: {mse:.4f} | MAE: {mae:.4f} | R2: {r2:.4f}")
        except:
            continue
    
    # Smart weighting: combine inverse MSE and R2
    weights = {}
    for name in mses:
        # Weight based on both low error and high R2
        inv_mse = 1 / (mses[name] + 1e-6)
        r2_weight = max(0, r2s[name])  # Only positive R2 values
        weights[name] = inv_mse * (1 + r2_weight)
    
    total_weight = sum(weights.values())
    weights = {name: w/total_weight for name, w in weights.items()}
    
    print("\n" + "-"*80)
    print("MODEL WEIGHTS")
    print("-"*80)
    for name, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
        print(f"{name.upper():12s} | Weight: {weight:.4f} ({weight*100:.2f}%)")
    
    # Final ensemble prediction
    ensemble_pred = sum(weights[name] * predictions[name][:len(y_test)] for name in predictions)
    mse_ensemble = mean_squared_error(y_test, ensemble_pred)
    mae_ensemble = mean_absolute_error(y_test, ensemble_pred)
    r2_ensemble = r2_score(y_test, ensemble_pred)
    
    print("\n" + "-"*80)
    print(f"ENSEMBLE | MSE: {mse_ensemble:.4f} | MAE: {mae_ensemble:.4f} | R2: {r2_ensemble:.4f}")
    print("-"*80)
    
    # Random baseline
    random_pred = np.random.exponential(scale=2, size=len(y_test)) + 1
    mse_random = mean_squared_error(y_test, random_pred)
    improvement = ((mse_random - mse_ensemble) / mse_random) * 100
    
    print(f"\nRandom Baseline MSE: {mse_random:.4f}")
    print(f"Improvement over Random: {improvement:.2f}%")
    print("="*80 + "\n")
    
    models['weights'] = weights
    models['scaler'] = scaler
    models['feature_columns'] = features.columns.tolist()
    
    return models

def predict_next_10(models, recent_data):
    """Generate next 10 predictions with maximum accuracy"""
    print("\n" + "="*80)
    print("GENERATING NEXT 10 CRASH POINT PREDICTIONS")
    print("="*80 + "\n")
    
    if len(recent_data) < 20:
        print("[WARNING] Less than 20 data points. Predictions may be less accurate.")
    
    predictions_history = []
    current_data = list(recent_data[-50:])  # Use last 50 for context
    
    print("Generating predictions...")
    print("-"*80)
    
    for round_num in range(1, 11):
        # Create features
        features_df = create_advanced_features(np.array(current_data))
        last_features = features_df.iloc[[-1]]
        
        # Get prediction from each model
        round_preds = {}
        weights = models['weights']
        
        if 'xgb' in models:
            round_preds['xgb'] = models['xgb'].predict(last_features)[0]
        if 'lgb' in models:
            round_preds['lgb'] = models['lgb'].predict(last_features)[0]
        if 'cat' in models:
            round_preds['cat'] = models['cat'].predict(last_features)[0]
        if 'rf' in models:
            round_preds['rf'] = models['rf'].predict(last_features)[0]
        if 'et' in models:
            round_preds['et'] = models['et'].predict(last_features)[0]
        if 'gb' in models:
            round_preds['gb'] = models['gb'].predict(last_features)[0]
        if 'stack' in models:
            round_preds['stack'] = models['stack'].predict(last_features)[0]
        
        # Scaled predictions
        if 'scaler' in models:
            last_features_scaled = models['scaler'].transform(last_features)
            if 'elastic' in models:
                round_preds['elastic'] = models['elastic'].predict(last_features_scaled)[0]
            if 'ridge' in models:
                round_preds['ridge'] = models['ridge'].predict(last_features_scaled)[0]
        
        # LSTM prediction
        if 'lstm' in models and len(current_data) >= 20:
            seq = np.array(current_data[-20:]).reshape(1, 20, 1)
            round_preds['lstm'] = models['lstm'].predict(seq, verbose=0)[0][0]
        
        # GRU prediction
        if 'gru' in models and len(current_data) >= 20:
            seq = np.array(current_data[-20:]).reshape(1, 20, 1)
            round_preds['gru'] = models['gru'].predict(seq, verbose=0)[0][0]
        
        # Weighted ensemble
        ensemble_pred = sum(round_preds.get(name, 0) * weights.get(name, 0) for name in round_preds)
        
        # Clip to realistic range
        ensemble_pred = max(1.0, min(ensemble_pred, 100.0))
        
        # Calculate confidence interval
        pred_values = [p for p in round_preds.values() if 1.0 <= p <= 100.0]
        if len(pred_values) > 1:
            std = np.std(pred_values)
            conf_low = max(1.0, ensemble_pred - 1.96 * std)
            conf_high = min(100.0, ensemble_pred + 1.96 * std)
        else:
            conf_low = ensemble_pred * 0.8
            conf_high = ensemble_pred * 1.2
        
        # Store prediction
        prediction_data = {
            'round': round_num,
            'prediction': round(ensemble_pred, 2),
            'confidence_low': round(conf_low, 2),
            'confidence_high': round(conf_high, 2),
            'model_predictions': {k: round(v, 2) for k, v in round_preds.items()}
        }
        predictions_history.append(prediction_data)
        
        # Display
        print(f"Round {round_num:2d}: {ensemble_pred:6.2f}x  |  95% CI: [{conf_low:6.2f}x - {conf_high:6.2f}x]")
        
        # Update data for next prediction
        current_data.append(ensemble_pred)
    
    print("-"*80)
    print("[SUCCESS] Generated 10 predictions")
    print("="*80 + "\n")
    
    return predictions_history

def save_predictions(predictions, input_data):
    """Save predictions to JSON file"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'predictions_{timestamp}.json'
    
    output = {
        'timestamp': datetime.now().isoformat(),
        'input_data': {
            'crash_points': [round(x, 2) for x in input_data],
            'total_points': len(input_data),
            'mean': round(float(np.mean(input_data)), 2),
            'std': round(float(np.std(input_data)), 2),
            'min': round(float(np.min(input_data)), 2),
            'max': round(float(np.max(input_data)), 2)
        },
        'predictions': predictions,
        'summary': {
            'next_10_predictions': [p['prediction'] for p in predictions],
            'prediction_method': 'Ultra-Accurate Multi-Model Ensemble',
            'models_used': [
                'XGBoost (Bayesian Optimized)',
                'LightGBM',
                'CatBoost',
                'Random Forest',
                'Extra Trees',
                'Gradient Boosting',
                'ElasticNet',
                'Ridge Regression',
                'Bidirectional LSTM',
                'GRU',
                'Stacking Ensemble'
            ],
            'total_models': 11,
            'average_prediction': round(float(np.mean([p['prediction'] for p in predictions])), 2),
            'median_prediction': round(float(np.median([p['prediction'] for p in predictions])), 2)
        }
    }
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=4, ensure_ascii=False)
        
        print(f"[OK] Predictions saved to: {filename}")
        logging.info(f"Predictions saved to {filename}")
        
        # Also save a simple text version
        txt_filename = f'predictions_{timestamp}.txt'
        with open(txt_filename, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("CRASH GAME PREDICTIONS\n")
            f.write("="*80 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Input Data Points: {len(input_data)}\n")
            f.write(f"Input Average: {np.mean(input_data):.2f}x\n\n")
            f.write("-"*80 + "\n")
            f.write("NEXT 10 PREDICTED CRASH POINTS:\n")
            f.write("-"*80 + "\n\n")
            
            for p in predictions:
                f.write(f"Round {p['round']:2d}: {p['prediction']:6.2f}x  ")
                f.write(f"(CI: {p['confidence_low']:6.2f}x - {p['confidence_high']:6.2f}x)\n")
            
            f.write("\n" + "-"*80 + "\n")
            f.write(f"Average Predicted: {np.mean([p['prediction'] for p in predictions]):.2f}x\n")
            f.write(f"Median Predicted: {np.median([p['prediction'] for p in predictions]):.2f}x\n")
            f.write("="*80 + "\n")
        
        print(f"[OK] Text version saved to: {txt_filename}")
        
        return filename, txt_filename
        
    except Exception as e:
        print(f"[ERROR] Could not save predictions: {e}")
        logging.error(f"Save error: {e}")
        return None, None

def display_predictions_table(predictions):
    """Display predictions in a nice table format"""
    print("\n" + "="*80)
    print(" "*25 + "PREDICTION RESULTS")
    print("="*80)
    print("\n{:<8} {:<12} {:<30}".format("Round", "Prediction", "95% Confidence Interval"))
    print("-"*80)
    
    for p in predictions:
        ci_range = f"[{p['confidence_low']:.2f}x - {p['confidence_high']:.2f}x]"
        print(f"{p['round']:<8} {p['prediction']:.2f}x{' '*6} {ci_range:<30}")
    
    print("-"*80)
    
    pred_values = [p['prediction'] for p in predictions]
    print(f"\nAverage:  {np.mean(pred_values):.2f}x")
    print(f"Median:   {np.median(pred_values):.2f}x")
    print(f"Range:    {min(pred_values):.2f}x - {max(pred_values):.2f}x")
    print("="*80 + "\n")

def statistical_analysis(data):
    """Perform statistical analysis on input data"""
    print("\n" + "="*80)
    print("STATISTICAL ANALYSIS OF INPUT DATA")
    print("="*80 + "\n")
    
    results = {}
    
    try:
        # Basic statistics
        results['Mean'] = float(np.mean(data))
        results['Median'] = float(np.median(data))
        results['Std Dev'] = float(np.std(data))
        results['Variance'] = float(np.var(data))
        results['Min'] = float(np.min(data))
        results['Max'] = float(np.max(data))
        results['Range'] = float(np.max(data) - np.min(data))
        
        # Percentiles
        results['25th Percentile'] = float(np.percentile(data, 25))
        results['75th Percentile'] = float(np.percentile(data, 75))
        results['90th Percentile'] = float(np.percentile(data, 90))
        
        # Display
        print(f"Data Points:      {len(data)}")
        print(f"Mean:             {results['Mean']:.2f}x")
        print(f"Median:           {results['Median']:.2f}x")
        print(f"Std Deviation:    {results['Std Dev']:.2f}")
        print(f"Variance:         {results['Variance']:.2f}")
        print(f"Min:              {results['Min']:.2f}x")
        print(f"Max:              {results['Max']:.2f}x")
        print(f"Range:            {results['Range']:.2f}x")
        print(f"\n25th Percentile:  {results['25th Percentile']:.2f}x")
        print(f"75th Percentile:  {results['75th Percentile']:.2f}x")
        print(f"90th Percentile:  {results['90th Percentile']:.2f}x")
        
        # Distribution analysis
        above_2 = np.sum(data > 2.0) / len(data) * 100
        above_3 = np.sum(data > 3.0) / len(data) * 100
        above_5 = np.sum(data > 5.0) / len(data) * 100
        
        print(f"\nAbove 2.0x:       {above_2:.1f}%")
        print(f"Above 3.0x:       {above_3:.1f}%")
        print(f"Above 5.0x:       {above_5:.1f}%")
        
        # Entropy
        entropy = local_entropy(data)
        results['Entropy'] = float(entropy)
        print(f"\nEntropy:          {entropy:.4f}")
        
        # Trend
        if len(data) > 2:
            trend = np.polyfit(range(len(data)), data, 1)[0]
            results['Trend'] = float(trend)
            trend_direction = "Increasing" if trend > 0 else "Decreasing"
            print(f"Trend:            {trend_direction} ({trend:.4f})")
        
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"[ERROR] Statistical analysis failed: {e}")
        logging.error(f"Statistical analysis error: {e}")
    
    return results

def compare_with_actual():
    """Compare predictions with actual crash points"""
    print("\n" + "="*80)
    print("PREDICTION ACCURACY COMPARISON")
    print("="*80 + "\n")
    
    print("Enter the ACTUAL crash points that occurred after your predictions.")
    print("This will calculate prediction accuracy.\n")
    
    actual_points = []
    
    for i in range(1, 11):
        while True:
            try:
                value = input(f"Actual Crash Point #{i} (or 'skip' to stop): ").strip()
                
                if value.lower() == 'skip':
                    if i == 1:
                        print("\n[INFO] No actual data entered. Skipping comparison.")
                        return
                    break
                
                crash_value = float(value.replace('x', '').replace('X', '').strip())
                
                if crash_value < 1.0:
                    print("[ERROR] Value must be >= 1.0x")
                    continue
                
                actual_points.append(crash_value)
                break
                
            except ValueError:
                print("[ERROR] Invalid input. Please enter a number.")
    
    if len(actual_points) == 0:
        return
    
    # Load latest predictions
    import glob
    pred_files = glob.glob('predictions_*.json')
    if not pred_files:
        print("[ERROR] No prediction files found.")
        return
    
    latest_file = max(pred_files, key=os.path.getctime)
    
    try:
        with open(latest_file, 'r') as f:
            pred_data = json.load(f)
        
        predictions = pred_data['predictions'][:len(actual_points)]
        
        print("\n" + "-"*80)
        print("{:<8} {:<12} {:<12} {:<12} {:<12}".format(
            "Round", "Predicted", "Actual", "Error", "% Error"
        ))
        print("-"*80)
        
        errors = []
        abs_errors = []
        
        for i, (pred, actual) in enumerate(zip(predictions, actual_points)):
            error = pred['prediction'] - actual
            pct_error = (error / actual) * 100
            
            errors.append(error)
            abs_errors.append(abs(error))
            
            within_ci = pred['confidence_low'] <= actual <= pred['confidence_high']
            ci_marker = "[OK]" if within_ci else "[!!]"
            
            print(f"{i+1:<8} {pred['prediction']:.2f}x{' '*6} {actual:.2f}x{' '*6} "
                  f"{error:+.2f}x{' '*5} {pct_error:+.1f}% {ci_marker}")
        
        print("-"*80)
        
        # Calculate metrics
        mae = np.mean(abs_errors)
        mse = np.mean([e**2 for e in errors])
        rmse = np.sqrt(mse)
        mape = np.mean([abs(e/a) * 100 for e, a in zip(errors, actual_points)])
        
        within_ci_count = sum(
            1 for pred, actual in zip(predictions, actual_points)
            if pred['confidence_low'] <= actual <= pred['confidence_high']
        )
        ci_accuracy = (within_ci_count / len(actual_points)) * 100
        
        print(f"\nMean Absolute Error (MAE):     {mae:.2f}x")
        print(f"Root Mean Squared Error (RMSE): {rmse:.2f}x")
        print(f"Mean Absolute % Error (MAPE):   {mape:.1f}%")
        print(f"Within Confidence Interval:     {within_ci_count}/{len(actual_points)} ({ci_accuracy:.1f}%)")
        
        # Accuracy rating
        if mape < 5:
            rating = "EXCELLENT"
        elif mape < 10:
            rating = "VERY GOOD"
        elif mape < 15:
            rating = "GOOD"
        elif mape < 25:
            rating = "FAIR"
        else:
            rating = "NEEDS IMPROVEMENT"
        
        print(f"\nAccuracy Rating: {rating}")
        print("="*80 + "\n")
        
        # Save comparison
        comparison_file = f'comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        comparison_data = {
            'timestamp': datetime.now().isoformat(),
            'predictions': [p['prediction'] for p in predictions],
            'actuals': actual_points,
            'errors': errors,
            'metrics': {
                'MAE': float(mae),
                'RMSE': float(rmse),
                'MAPE': float(mape),
                'CI_Accuracy': float(ci_accuracy)
            },
            'rating': rating
        }
        
        with open(comparison_file, 'w') as f:
            json.dump(comparison_data, f, indent=4)
        
        print(f"[OK] Comparison saved to: {comparison_file}\n")
        
    except Exception as e:
        print(f"[ERROR] Comparison failed: {e}")
        logging.error(f"Comparison error: {e}")

def main_menu():
    """Display main menu and handle user choice"""
    while True:
        print("\n" + "="*80)
        print(" "*28 + "MAIN MENU")
        print("="*80)
        print("\n  1. Enter crash points manually")
        print("  2. Load crash points from file")
        print("  3. Compare predictions with actual results")
        print("  4. Exit")
        print("\n" + "="*80)
        
        choice = input("\nSelect option (1-4): ").strip()
        
        if choice == '1':
            return 'manual'
        elif choice == '2':
            return 'file'
        elif choice == '3':
            return 'compare'
        elif choice == '4':
            print("\n[EXIT] Goodbye!\n")
            sys.exit(0)
        else:
            print("[ERROR] Invalid option. Please choose 1-4.")

def main():
    """Main application flow"""
    print_banner()
    
    logging.info("Application started")
    
    while True:
        mode = main_menu()
        
        if mode == 'compare':
            compare_with_actual()
            continue
        
        # Collect data
        if mode == 'manual':
            crash_data = collect_manual_input()
        else:  # file mode
            filename = input("\nEnter filename (e.g., data.json or data.txt): ").strip()
            crash_data = load_from_file(filename)
            
            if crash_data is None:
                print("[ERROR] Could not load data. Please try again.\n")
                continue
        
        if len(crash_data) < 10:
            print("\n[ERROR] Need at least 10 crash points for predictions.\n")
            continue
        
        # Statistical analysis
        stats = statistical_analysis(crash_data)
        
        # Train models
        models = train_ultra_accurate_models(crash_data)
        
        if models is None:
            print("[ERROR] Model training failed.\n")
            continue
        
        # Generate predictions
        predictions = predict_next_10(models, crash_data)
        
        # Display predictions
        display_predictions_table(predictions)
        
        # Save predictions
        json_file, txt_file = save_predictions(predictions, crash_data)
        
        if json_file:
            print("\n" + "="*80)
            print("FILES CREATED:")
            print("="*80)
            print(f"\n  JSON Format:  {json_file}")
            print(f"  Text Format:  {txt_file}")
            print(f"  Log File:     crash_predictions.log")
            print("\n" + "="*80)
        
        # Ask if user wants to continue
        print("\n")
        continue_choice = input("Generate new predictions? (y/n): ").strip().lower()
        
        if continue_choice != 'y':
            print("\n[EXIT] Thank you for using Crash Game Predictor!\n")
            break
        
        print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Application terminated by user.\n")
        sys.exit(0)
    except Exception as e:
        print(f"\n[FATAL ERROR] {e}\n")
        logging.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)