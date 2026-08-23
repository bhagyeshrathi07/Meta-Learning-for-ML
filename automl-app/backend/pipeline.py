import pandas as pd
import numpy as np
import time
import os
import joblib
import re
import random

# Scikit-Learn & XGBoost
from sklearn.model_selection import train_test_split, RandomizedSearchCV, ParameterGrid
from sklearn.preprocessing import RobustScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, f1_score, roc_curve, auc, confusion_matrix, 
                             r2_score, mean_squared_error, mean_absolute_error, precision_score, recall_score)

# Models
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, SGDClassifier, SGDRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor

from monitor import ResourceMonitor

def clean_column_names(df):
    """Cleans column names for XGBoost compatibility"""
    regex = re.compile(r"\[|\]|<", re.IGNORECASE)
    df.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in df.columns]
    return df

def get_configs(task_type, n_rows, scale_pos_weight=1):
    """
    Merged Configs:
    - Uses 'Large Data' switching logic (SGD instead of SVM).
    - Uses Your 'class_weight=balanced' logic.
    - Uses Your 'scale_pos_weight' for XGBoost.
    - ENHANCED: Wider hyperparameter search spaces for better accuracy.
    """
    is_large_data = n_rows > 10000
    
    # --- SHARED PARAMS (Trees) ---
    rf_params = {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [None, 10, 20, 30],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4]
    }
    xgb_params = {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__learning_rate': [0.01, 0.05, 0.1, 0.2],
        'classifier__max_depth': [3, 6, 10],
        'classifier__subsample': [0.7, 0.8, 1.0],
        'classifier__colsample_bytree': [0.7, 0.8, 1.0]
    }

    if task_type == 'Classification':
        print(f"⚡ Classification Mode: {'High-Performance (SGD)' if is_large_data else 'High-Accuracy (SVM)'}")
        
        # SGD (Large Data)
        sgd_config = {
            'model': SGDClassifier(loss='modified_huber', class_weight='balanced', random_state=42, early_stopping=True),
            'params': {
                'classifier__alpha': [0.0001, 0.001, 0.01], 
                'classifier__penalty': ['l2', 'elasticnet'],
                'classifier__l1_ratio': [0.15, 0.5, 0.85] # Only used if elasticnet
            }
        }
        
        # SVM (Small Data) - Added Gamma and wider C
        svc_config = {
            'model': SVC(probability=True, class_weight='balanced', random_state=42),
            'params': {
                'classifier__C': [0.1, 1, 10, 100], 
                'classifier__kernel': ['linear', 'rbf'],
                'classifier__gamma': ['scale', 'auto']
            }
        }

        return {
            'Logistic Regression': {
                'model': LogisticRegression(solver='liblinear', class_weight='balanced', max_iter=1000), 
                'params': {'classifier__C': [0.01, 0.1, 1, 10, 100], 'classifier__penalty': ['l1', 'l2']}
            },
            'Random Forest': {
                'model': RandomForestClassifier(class_weight='balanced', random_state=42), 
                'params': rf_params
            },
            'Decision Tree': { 
                'model': DecisionTreeClassifier(class_weight='balanced', random_state=42),
                'params': {
                    'classifier__max_depth': [5, 10, 20, None],
                    'classifier__min_samples_split': [2, 5, 10],
                    'classifier__min_samples_leaf': [1, 2, 4]
                }
            },
            'SVM': sgd_config if is_large_data else svc_config,
            'KNN': {
                'model': KNeighborsClassifier(), 
                'params': {
                    'classifier__n_neighbors': [3, 5, 7, 9, 11], 
                    'classifier__weights': ['uniform', 'distance'],
                    'classifier__p': [1, 2] # 1=Manhattan, 2=Euclidean
                }
            },
            'XGBoost': {
                'model': XGBClassifier(eval_metric='logloss', scale_pos_weight=scale_pos_weight, use_label_encoder=False, random_state=42, nthread=1), 
                'params': xgb_params
            }
        }

    else: # Regression
        print(f"⚡ Regression Mode: {'High-Performance (SGD)' if is_large_data else 'High-Accuracy (SVR)'}")
        
        sgd_reg_config = {
            'model': SGDRegressor(random_state=42, early_stopping=True),
            'params': {
                'classifier__alpha': [0.0001, 0.001, 0.01], 
                'classifier__penalty': ['l2', 'elasticnet']
            }
        }

        svr_config = {
            'model': SVR(),
            'params': {
                'classifier__C': [0.1, 1, 10, 100], 
                'classifier__kernel': ['linear', 'rbf'],
                'classifier__gamma': ['scale', 'auto']
            }
        }

        return {
            'Linear Regression': {'model': LinearRegression(), 'params': {'classifier__fit_intercept': [True, False]}},
            'Decision Tree': {
                'model': DecisionTreeRegressor(random_state=42), 
                'params': {
                    'classifier__max_depth': [5, 10, 20, None],
                    'classifier__min_samples_split': [2, 5, 10],
                    'classifier__min_samples_leaf': [1, 2, 4]
                }
            },
            'Random Forest': {
                'model': RandomForestRegressor(random_state=42), 
                'params': rf_params
            },
            'XGBoost': {
                'model': XGBRegressor(random_state=42, nthread=1), 
                'params': xgb_params
            },
            'SVM': sgd_reg_config if is_large_data else svr_config
        }

def run_automl(filepath, target_column, selected_models=None, callback=None):
    if callback: callback(5, "📂 Loading dataset...")
    
    # Smart Parsing
    df = pd.read_csv(filepath, sep=None, engine='python')
    df = clean_column_names(df) # Regex Fix
    
    # --- ROBUST CLEANING ---
    if callback: callback(10, "🧹 Cleaning data...")
    
    # A. Drop IDs
    for col in df.columns:
        if 'id' in col.lower().split('_') or col.lower() == 'id':
             df = df.drop(columns=[col])

    df = df.dropna(axis=1, how='all')
    df = df.dropna(subset=[target_column])

    # B. Fix Numeric Strings & Drop Dates/High Cardinality
    for col in list(df.select_dtypes(include=['object']).columns):
        if col == target_column: continue
        
        # Try converting "$1,000" to 1000
        numeric_conversion = pd.to_numeric(df[col], errors='coerce')
        if numeric_conversion.notna().mean() > 0.8:
            df[col] = numeric_conversion
            continue

        # Drop Dates
        try:
            pd.to_datetime(df[col], errors='raise', utc=True)
            df = df.drop(columns=[col])
            continue
        except: pass

        # Drop High Cardinality (Names, Descriptions)
        if df[col].nunique() > 50 and (df[col].nunique() / len(df)) > 0.9:
            df = df.drop(columns=[col])

    # --- SAMPLING LOGIC (Crucial for Speed) ---
    MAX_ROWS = 100000
    if len(df) > MAX_ROWS:
        if callback: callback(12, f"⚠️ Downsampling {len(df)} rows to {MAX_ROWS}...")
        try:
            # Stratify if classification to keep imbalance ratio
            if df[target_column].nunique() < 20:
                df, _ = train_test_split(df, train_size=MAX_ROWS, stratify=df[target_column], random_state=42)
            else:
                df = df.sample(n=MAX_ROWS, random_state=42)
        except:
            df = df.sample(n=MAX_ROWS, random_state=42)

    # --- TASK DETECTION ---
    y = df[target_column]
    task_type = "Classification"
    if pd.api.types.is_numeric_dtype(y) and y.nunique() > 20:
        task_type = "Regression"
            
    if callback: callback(15, f"🔍 Task: {task_type} | Rows: {len(df)}")

    # --- PREPROCESSING ---
    X = df.drop(columns=[target_column])
    y = df[target_column]

    if task_type == "Classification" and (y.dtype == 'object' or isinstance(y.iloc[0], str)):
        le = LabelEncoder()
        y = le.fit_transform(y)

    # RobustScaler (better for outliers)
    num_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', RobustScaler())])
    cat_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(transformers=[
        ('num', num_transformer, X.select_dtypes(include=['number']).columns), 
        ('cat', cat_transformer, X.select_dtypes(exclude=['number']).columns)
    ])

    if task_type == "Classification":
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- IMBALANCE LOGIC ---
    scale_pos_weight = 1
    if task_type == "Classification" and len(np.unique(y)) == 2:
        negatives = np.sum(y_train == 0)
        positives = np.sum(y_train == 1)
        if positives > 0:
            scale_pos_weight = negatives / positives

    # --- TRAINING LOOP ---
    configs = get_configs(task_type, len(X_train), scale_pos_weight)
    results = []
    trained_models = {}
    
    models_to_run = []
    if selected_models:
        for m in selected_models:
            matched = next((k for k in configs.keys() if m in k or k in m), None)
            if matched: models_to_run.append(matched)
    else:
        models_to_run = list(configs.keys())
    
    total_models = len(models_to_run)
    current_model_idx = 0

    for name in models_to_run:
        config = configs[name]
        current_model_idx += 1
        progress_pct = 20 + int((current_model_idx / total_models) * 70)
        if callback: callback(progress_pct, f"🏋️ Training {name}...")

        clf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', config['model'])]) 

        monitor = ResourceMonitor()
        with monitor:
            # Calculate max possible combinations for this specific model
            max_combinations = len(ParameterGrid(config['params']))
            # Set n_iter to 5, OR the max_combinations if the grid is smaller
            current_n_iter = min(5, max_combinations)
            # Use n_jobs=1 for XGBoost on Windows
            n_jobs_param = 1 if 'XGBoost' in name else -1
        
        search = RandomizedSearchCV(
            clf, 
            config['params'], 
            n_iter=current_n_iter,
            cv=3, 
            n_jobs=n_jobs_param, 
            random_state=42
        )
        
        # --- 1. START TIMER ---
        start_time = time.time()

        # Monitor MUST wrap the actual training!
        with monitor:
            try:
                search.fit(X_train, y_train)
            except Exception as e:
                print(f"Model {name} failed: {e}")
                continue

        # --- 2. STOP TIMER ---
        end_time = time.time()
        training_duration = round(end_time - start_time, 2)

        best_model = search.best_estimator_
        trained_models[name] = best_model 
        y_pred = best_model.predict(X_test)

        metrics = {
            "Model": name,
            "Task Type": task_type,
            "Training Time (s)": training_duration,
            "Max RAM (MB)": round(monitor.max_ram, 2),
            "Max CPU (%)": round(monitor.max_cpu, 2),
            "Best Params": search.best_params_
        }

        if task_type == "Classification":
            metrics.update({
                "Accuracy": round(accuracy_score(y_test, y_pred), 4),
                "F1 Score": round(f1_score(y_test, y_pred, average='weighted'), 4),
                "Precision": round(precision_score(y_test, y_pred, average='weighted'), 4),
                "Recall": round(recall_score(y_test, y_pred, average='weighted'), 4),
                "ConfusionMatrix": confusion_matrix(y_test, y_pred).tolist()
            })
            # ROC Logic
            if len(np.unique(y)) == 2:
                try:
                    if hasattr(best_model.named_steps['classifier'], 'predict_proba'):
                        y_prob = best_model.predict_proba(X_test)[:, 1]
                        fpr, tpr, _ = roc_curve(y_test, y_prob)
                        metrics["AUC"] = round(auc(fpr, tpr), 4)
                        if len(fpr) < 20:
                            metrics["ROCData"] = [{"x": round(fpr[i], 3), "y": round(tpr[i], 3)} for i in range(len(fpr))]
                        else:
                            metrics["ROCData"] = [{"x": round(f,3), "y": round(t,3)} for f,t in zip(fpr[::5], tpr[::5])]
                            metrics["ROCData"].append({"x": round(fpr[-1], 3), "y": round(tpr[-1], 3)})
                    else:
                        metrics["AUC"] = "N/A (Linear Model)"
                        metrics["ROCData"] = []
                except: pass
        else:
            # Regression Metrics
            metrics.update({
                "R2 Score": round(r2_score(y_test, y_pred), 4),
                "RMSE": round(np.sqrt(mean_squared_error(y_test, y_pred)), 4),
                "MAE": round(mean_absolute_error(y_test, y_pred), 4)
            })
            
            # Scatter Plot Data
            indices = np.arange(len(y_test))
            np.random.shuffle(indices)
            sample_indices = indices[:200]
            scatter_data = []
            y_test_arr = np.array(y_test)
            for i in sample_indices:
                scatter_data.append({
                    "actual": float(round(y_test_arr[i], 2)),
                    "predicted": float(round(y_pred[i], 2))
                })
            metrics["ScatterData"] = scatter_data

        results.append(metrics)

    # Sort Results
    if task_type == "Classification":
        results.sort(key=lambda x: x.get('Accuracy', 0), reverse=True)
    else:
        results.sort(key=lambda x: x.get('R2 Score', -float('inf')), reverse=True)

    # Save Models (Absolute Path)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    for name, model in trained_models.items():
        joblib.dump(model, os.path.join(models_dir, name.replace(" ", "_") + ".pkl"))

    return results