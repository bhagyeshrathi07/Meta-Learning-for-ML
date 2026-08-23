def generate_training_script(model_name, best_params, task_type, target_col):
    param_str = ", ".join([f"{k.split('__')[1]}={v!r}" for k, v in best_params.items()])
    imports = ""
    model_code = ""
    
    if "Random Forest" in model_name:
        imports = f"from sklearn.ensemble import RandomForest{'Classifier' if task_type == 'Classification' else 'Regressor'}"
        model_code = f"model = RandomForest{'Classifier' if task_type == 'Classification' else 'Regressor'}({param_str})"
    elif "XGBoost" in model_name:
        imports = f"from xgboost import XGB{'Classifier' if task_type == 'Classification' else 'Regressor'}"
        model_code = f"model = XGB{'Classifier' if task_type == 'Classification' else 'Regressor'}({param_str})"
    elif "Logistic" in model_name or "Linear" in model_name:
        imports = f"from sklearn.linear_model import {'LogisticRegression' if task_type == 'Classification' else 'LinearRegression'}"
        model_code = f"model = {'LogisticRegression' if task_type == 'Classification' else 'LinearRegression'}({param_str})"
    elif "Decision Tree" in model_name:
        imports = f"from sklearn.tree import DecisionTree{'Classifier' if task_type == 'Classification' else 'Regressor'}"
        model_code = f"model = DecisionTree{'Classifier' if task_type == 'Classification' else 'Regressor'}({param_str})"
    elif "KNN" in model_name:
        imports = f"from sklearn.neighbors import {'KNeighborsClassifier' if task_type == 'Classification' else 'KNeighborsRegressor'}"
        model_code = f"model = {'KNeighborsClassifier' if task_type == 'Classification' else 'KNeighborsRegressor'}({param_str})"
    elif "SVM" in model_name:
        # DETECT FLAVOR: Check if we tuned 'alpha' (SGD) or 'C' (Standard SVM)
        is_sgd = any('alpha' in k for k in best_params.keys())
        
        if is_sgd:
            # SGD version (larget Data)
            imports = f"from sklearn.linear_model import SGD{'Classifier' if task_type == 'Classification' else 'Regressor'}"
            # We must explicitly add the loss function we chose in pipeline.py
            if task_type == 'Classification':
                # Add loss parameter for classification
                loss_param = "loss='modified_huber', "
                model_code = f"model = SGDClassifier({loss_param}{param_str})"
            else:
                # Add loss parameter for regression
                loss_param = "loss='squared_error', "
                model_code = f"model = SGDRegressor({loss_param}{param_str})"
        else:
            # Standard SVC/SVR version
            imports = f"from sklearn.svm import {'SVC' if task_type == 'Classification' else 'SVR'}"
            # Ensure probability is on for SVC to match pipeline behavior
            if task_type == 'Classification':
                prob_param = "probability=True, "
                model_code = f"model = SVC({prob_param}{param_str})"
            else:
                # For SVR, no extra params needed (epsilon is in best_params)
                model_code = f"model = SVR({param_str})"

    metrics_import = ""
    eval_code = ""
    if task_type == "Classification":
        metrics_import = "from sklearn.metrics import accuracy_score, classification_report"
        eval_code = 'print("Accuracy:", accuracy_score(y_test, y_pred))\nprint(classification_report(y_test, y_pred))'
    else:
        metrics_import = "from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error"
        eval_code = 'print("R2 Score:", r2_score(y_test, y_pred))\nprint("MAE:", mean_absolute_error(y_test, y_pred))\nprint("RMSE:", mean_squared_error(y_test, y_pred, squared=False))'

    script = f"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
{imports}
{metrics_import}

# 1. Load Data
# TODO: Update this path
df = pd.read_csv('your_dataset.csv') 
target_col = '{target_col}'

# 2. Preprocessing
X = df.drop(columns=[target_col])
y = df[target_col]

num_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', RobustScaler())])
cat_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(transformers=[
    ('num', num_transformer, X.select_dtypes(include=['number']).columns),
    ('cat', cat_transformer, X.select_dtypes(exclude=['number']).columns)
])

# 3. Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train
{model_code}
clf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
clf.fit(X_train, y_train)

# 5. Evaluate
y_pred = clf.predict(X_test)
{eval_code}
"""
    return script