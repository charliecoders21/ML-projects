import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
import joblib
import os

# ---------------------------
# Load and clean data
# ---------------------------
loan_approve_pred_df = pd.read_csv('data/loan_approval_dataset.csv')

# Clean column names (remove extra spaces)
loan_approve_pred_df.columns = loan_approve_pred_df.columns.str.strip()

# Drop ID column if present
if 'loan_id' in loan_approve_pred_df.columns:
    loan_approve_pred_df.drop(columns=['loan_id'], inplace=True)

# ---------------------------
# Feature Engineering: Total Asset Value
# ---------------------------
asset_cols = ['residential_assets_value', 'commercial_assets_value', 'luxury_assets_value', 'bank_asset_value']

# Fill missing values with 0 (optional, based on your data quality)
loan_approve_pred_df[asset_cols] = loan_approve_pred_df[asset_cols].fillna(0)

# Create new feature
loan_approve_pred_df['total_asset_value'] = loan_approve_pred_df[asset_cols].sum(axis=1)

# ---------------------------
# Encode categorical columns
# ---------------------------
categorized_col = ['education', 'self_employed', 'loan_status']
label_encoders = {}

for col in categorized_col:
    le = LabelEncoder()
    loan_approve_pred_df[col] = le.fit_transform(loan_approve_pred_df[col])
    label_encoders[col] = le  # Save encoder for later use

# ---------------------------
# Split dataset
# ---------------------------
X = loan_approve_pred_df.drop('loan_status', axis=1)
Y = loan_approve_pred_df['loan_status']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# ---------------------------
# Pipeline setup
# ---------------------------
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', XGBClassifier(eval_metric='logloss', use_label_encoder=False, random_state=42))
])

param_grid = {
    'classifier__n_estimators': [100, 150],
    'classifier__learning_rate': [0.05, 0.1],
    'classifier__max_depth': [3, 5],
    'classifier__subsample': [0.8, 1.0],
    'classifier__reg_lambda': [1, 5],
    'classifier__reg_alpha': [0, 1]
}

# ---------------------------
# GridSearchCV training
# ---------------------------
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1', n_jobs=-1)
grid_search.fit(X_train, Y_train)

# ---------------------------
# Evaluation
# ---------------------------
y_pred = grid_search.predict(X_test)

print("Best Params:", grid_search.best_params_)
print("Accuracy:", accuracy_score(Y_test, y_pred))
print("Classification Report:\n", classification_report(Y_test, y_pred))

# ---------------------------
# Save model and metadata
# ---------------------------
os.makedirs("models", exist_ok=True)
joblib.dump(grid_search.best_estimator_, "models/loan_approval_predict.joblib")
joblib.dump(X_train.columns.tolist(), "models/model_features.pkl")
joblib.dump(label_encoders, "models/label_encoders.pkl")

print("✅ Model, features, and encoders saved successfully.")