import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import VotingRegressor
import joblib

# ---------------------------
# 1️⃣ Load & Clean Data
# ---------------------------
salary_df = pd.read_csv('data/Salary_Data.csv')

# Fill missing values
for col in ['Age', 'Years of Experience', 'Salary']:
 salary_df[col].fillna(salary_df[col].median(), inplace=True)
for col in ['Gender', 'Education Level', 'Job Title']:
 salary_df[col].fillna(salary_df[col].mode()[0], inplace=True)

salary_df.drop_duplicates(inplace=True)

# Encode categorical variables
categorical_features = ['Gender', 'Age']
for feature in categorical_features:
 
 salary_df.drop(columns=feature, inplace=True)
label_enc = LabelEncoder()
salary_df['Job Title']= label_enc.fit_transform(salary_df['Job Title'])
salary_df['Education Level'] = label_enc.fit_transform(salary_df['Education Level'])
# Log-transform skewed features
salary_df['Years of Experience'] = np.log1p(salary_df['Years of Experience'])
y_log = np.log1p(salary_df['Salary'])
X = salary_df.drop('Salary', axis=1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
 X, y_log, test_size=0.2, random_state=42
)

# ---------------------------
# 2️⃣ Safe exponential function
# ---------------------------
def safe_expm1(x):
 x = np.nan_to_num(x, nan=0.0, posinf=20, neginf=-5)
 x = np.clip(x, -5, 20)
 return np.expm1(x)

# ---------------------------
# 3️⃣ Evaluation function
# ---------------------------
def evaluate_model(name, model, X_test, y_test):
 y_pred_log = model.predict(X_test)
 y_pred = safe_expm1(y_pred_log)
 y_true = safe_expm1(y_test)
 
 mse = mean_squared_error(y_true, y_pred)
 mae = mean_absolute_error(y_true, y_pred)
 r2 = r2_score(y_true, y_pred)
 adj_r2 = 1 - (1 - r2) * (len(y_true)-1) / (len(y_true)-X_test.shape[1]-1)
 
 print(f"\n{name} Results:")
 print(f" MSE: {mse:.2f}, MAE: {mae:.2f}, R²: {r2:.4f}, Adj R²: {adj_r2:.4f}")
 print(" Sample Predictions:")
 for actual, pred in zip(y_true[:5], y_pred[:5]):
    print(f" Actual: {actual:.2f}, Predicted: {pred:.2f}")
 
 return r2

# ---------------------------
# 4️⃣ Define Models
# ---------------------------
xgb_model = XGBRegressor(
 n_estimators=1000,
 learning_rate=0.05,
 max_depth=5,
 subsample=0.8,
 colsample_bytree=0.8,
 random_state=42
)

ridge_model = Ridge(alpha=1)
elastic_model = ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000)

# Ensemble: combine XGB + Ridge + ElasticNet
ensemble = VotingRegressor([
 ('xgb', xgb_model),
 ('ridge', ridge_model),
 ('elastic', elastic_model)
])

# ---------------------------
# 5️⃣ Train Ensemble
# ---------------------------
ensemble.fit(X_train, y_train)

# ---------------------------
# 6️⃣ Evaluate
# ---------------------------
evaluate_model("Ensemble (XGB + Ridge + ElasticNet)", ensemble, X_test, y_test)

# ---------------------------
# 7️⃣ Save Model
joblib.dump(ensemble, 'salary_prediction_model.joblib')
# ---------------------------

