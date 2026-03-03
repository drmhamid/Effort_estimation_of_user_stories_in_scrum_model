# train_and_save_model.py
# Trains ET, RF, LR, XGB using specified 11 features on ALL data
# and saves the final fitted pipelines for deployment.

import pandas as pd
import numpy as np
# No train_test_split needed if training on all data
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
# No metrics needed here if not evaluating during training
# from sklearn.metrics import mean_squared_error, mean_absolute_error

# --- Import Required Regressors ---
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
import xgboost as xgb
# ----------------------------------

import joblib
import os
import traceback
import warnings
import sklearn

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning, message='.*feature_names mismatch.*')

print(f"Using scikit-learn version: {sklearn.__version__}")
print(f"Using xgboost version: {xgb.__version__}")
print("--- Final Ensemble Model Training & Saving Script (ET, RF, LR, XGB) ---")

# --- Configuration ---
DATASET_PATH = "FINALIZED_DATASET.csv" # Ensure this is correct
MODEL_DIR = "models" # Folder to save models

# Define filenames for the FINAL saved pipelines
# Use consistent names that app.py will expect
ET_PIPELINE_FILENAME = os.path.join(MODEL_DIR, "et_pipeline_final.joblib")
RF_PIPELINE_FILENAME = os.path.join(MODEL_DIR, "rf_pipeline_final.joblib")
LR_PIPELINE_FILENAME = os.path.join(MODEL_DIR, "lr_pipeline_final.joblib")
XGB_PIPELINE_FILENAME = os.path.join(MODEL_DIR, "xgb_pipeline_final.joblib")

os.makedirs(MODEL_DIR, exist_ok=True)
print(f"Models will be saved in: {MODEL_DIR}")

# --- 1. Load Data & Initial Cleaning ---
try:
    df = pd.read_csv(DATASET_PATH)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns.")
    df.columns = df.columns.str.strip()

    # Handle NaNs in specific categorical columns
    if 'relatedtechnologies' in df.columns: df['relatedtechnologies'].fillna('None', inplace=True)
    if 'dbms' in df.columns: df['dbms'].fillna('None', inplace=True)

except FileNotFoundError: print(f"FATAL Error: Dataset file '{DATASET_PATH}' not found."); exit()
except Exception as e: print(f"FATAL Error loading data: {e}"); exit()

# --- 2. Define Features (X) and Target (y) ---
# Using the exact 11 features from your final evaluation script
base_features = [
    'Size', 'Complexity', 'Priority', 'Noftasks', 'developmenttype',
    'externalhardware', 'relatedtechnologies', 'dbms', 'Requirement Volatility',
    'Teammembers', 'PL'
]
target_column = 'Effort'

print(f"\nUsing these {len(base_features)} features for final models: {base_features}")
print(f"Target column: {target_column}")

# Check columns, map text, convert to numeric, handle NaNs
required_cols = base_features + [target_column]
missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols: print(f"FATAL Error: Missing columns: {missing_cols}"); exit()

print("\nApplying text-to-numeric mapping (if needed)...")
size_map = {'Very Low': 1, 'Low': 2, 'Medium': 3, 'High': 4, 'Highest': 5}
complexity_map = {'Very Low': 1, 'Low': 2, 'Medium': 3, 'High': 4, 'Highest': 5}
req_volatility_map = {'High': 1, 'Medium': 2, 'Low': 3}
cols_to_map = {'Size': size_map, 'Complexity': complexity_map, 'Requirement Volatility': req_volatility_map}
for col, mapping in cols_to_map.items():
    if col in df.columns and df[col].dtype == 'object':
         print(f" - Mapping column '{col}'")
         df[col] = df[col].astype(str).replace(mapping)

print("Converting columns to numeric...")
cols_to_convert_numeric = ['Size', 'Complexity', 'Priority', 'Noftasks', 'externalhardware', 'Requirement Volatility', 'Teammembers', target_column]
for col in cols_to_convert_numeric:
    if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop rows with NaNs in features OR target before selecting X and y
df.dropna(subset=required_cols, inplace=True)
print(f"Data shape after NaN drop: {df.shape}")
if df.empty: print("FATAL Error: No data left after cleaning."); exit()

# Select FINAL X and y from the cleaned data
X = df[base_features].copy()
y = df[target_column]
print(f"Using {len(X)} final samples for training.")

# --- 3. Identify Feature Types for Preprocessor ---
numerical_cols = X.select_dtypes(include=np.number).columns.tolist()
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
print(f"\nFinal Numerical columns for preprocessor: {numerical_cols}")
print(f"Final Categorical columns for preprocessor: {categorical_cols}")
defined_cols = set(numerical_cols + categorical_cols); base_set = set(base_features)
if defined_cols != base_set: print(f"FATAL Error: Feature type mismatch: {base_set - defined_cols}"); exit()

# --- 4. Define Preprocessing Steps --- # NO SPLIT NEEDED
numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='missing')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numerical_cols), ('cat', categorical_transformer, categorical_cols)], remainder='passthrough')
print("\nPreprocessor defined.")

# --- 5. Define Individual Models ---
print("Defining base models...")
# Use the same parameters as your final evaluation script
et_model = ExtraTreesRegressor(n_estimators=200, random_state=42, n_jobs=-1) # Matched n_estimators=200 from eval script
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
lr_model = LinearRegression()
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42, n_jobs=-1)

# --- 6. Create Full Pipelines ---
print("Creating full pipelines...")
et_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', et_model)])
rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', rf_model)])
lr_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', lr_model)])
xgb_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', xgb_model)])

# --- 7. Train All Pipelines on the ENTIRE Dataset ---
print("\n--- Training Final Pipelines on ALL Data ---")
pipelines_to_train = {
    "Extra Trees": et_pipeline,
    "Random Forest": rf_pipeline,
    "Linear Regression": lr_pipeline,
    "XGBoost": xgb_pipeline
}
try:
    # Train using the full X and y
    for name, pipe in pipelines_to_train.items():
        print(f"Training {name} pipeline...")
        pipe.fit(X, y)
    print("--- Training Complete ---")
except Exception as e:
    print(f"FATAL Error during training: {e}"); print(traceback.format_exc()); exit()

# --- 8. Save Fitted Pipelines ---
print("\n--- Saving Trained Final Pipelines ---")
pipeline_files = {
    et_pipeline: ET_PIPELINE_FILENAME,
    rf_pipeline: RF_PIPELINE_FILENAME,
    lr_pipeline: LR_PIPELINE_FILENAME,
    xgb_pipeline: XGB_PIPELINE_FILENAME
}
try:
    for pipe, filename in pipeline_files.items():
        joblib.dump(pipe, filename)
        print(f" - Saved: {filename}")
except Exception as e:
    print(f"Error saving pipelines: {e}")

print("\n--- Final Model Training & Saving Script Finished ---")