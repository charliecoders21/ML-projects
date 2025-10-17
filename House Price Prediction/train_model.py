# ==========================================
# 📘 HOUSING PRICE PREDICTION - VERSION 5
# (LightGBM + Target Encoding + Pipeline + KFold)
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from lightgbm import LGBMRegressor
from category_encoders import TargetEncoder

# -----------------------------
# 1️⃣ Load Dataset
# -----------------------------
housing_df = pd.read_csv("data/Housing.csv")

# -----------------------------
# 2️⃣ Feature Engineering
# -----------------------------
housing_df['area_per_bedroom'] = housing_df['area'] / (housing_df['bedrooms'] + 1)
housing_df['area_stories'] = housing_df['area'] * housing_df['stories']
housing_df['bed_bath'] = housing_df['bedrooms'] * housing_df['bathrooms']
housing_df['amenities_score'] = housing_df[['mainroad','guestroom','basement',
                                            'hotwaterheating','airconditioning','prefarea']].replace({'yes':1, 'no':0}).sum(axis=1)
housing_df['parking_pref'] = housing_df['parking'] * housing_df['prefarea'].map({'yes':1, 'no':0})
housing_df['lux_score'] = housing_df['airconditioning'].map({'yes':1, 'no':0}) + housing_df['hotwaterheating'].map({'yes':1, 'no':0})

# -----------------------------
# 3️⃣ Log Transform Target
# -----------------------------
housing_df['log_price'] = np.log1p(housing_df['price'])

# -----------------------------
# 4️⃣ Define Features/Target
# -----------------------------
X = housing_df.drop(['price', 'log_price'], axis=1)
y = housing_df['log_price']

# -----------------------------
# 5️⃣ Column Definitions
# -----------------------------
num_features = ['area','bedrooms','bathrooms','stories','parking',
                'area_per_bedroom','area_stories','bed_bath',
                'amenities_score','parking_pref','lux_score']
cat_features = ['mainroad','guestroom','basement','hotwaterheating',
                'airconditioning','prefarea','furnishingstatus']

# -----------------------------
# 6️⃣ Preprocessing Pipeline
# -----------------------------
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_features),
    ('cat', TargetEncoder(), cat_features)
])

# -----------------------------
# 7️⃣ Hyperparameter Tuning
# -----------------------------
param_dist = {
    'n_estimators': [300, 500, 800],
    'learning_rate': [0.01, 0.03, 0.05, 0.1],
    'max_depth': [4, 5, 6, 8],
    'num_leaves': [20, 31, 50],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'reg_lambda': [0.1, 1, 10],
    'reg_alpha': [0, 0.5, 1]
}

pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('model', LGBMRegressor(random_state=42))
])

search = RandomizedSearchCV(
    pipeline,
    param_distributions={'model__' + k: v for k, v in param_dist.items()},
    n_iter=20,
    cv=5,
    scoring='neg_root_mean_squared_error',
    random_state=42,
    n_jobs=-1,
    verbose=1
)
search.fit(X, y)

best_params = {k.replace('model__', ''): v for k, v in search.best_params_.items()}
print("\n✅ Best Hyperparameters:", best_params)

# -----------------------------
# 8️⃣ Final Model Training
# -----------------------------
final_pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('model', LGBMRegressor(**best_params, random_state=42))
])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
final_pipeline.fit(X_train, y_train)

# -----------------------------
# 9️⃣ Predictions & Evaluation
# -----------------------------
y_pred_log = final_pipeline.predict(X_test)
y_pred = np.expm1(y_pred_log)
y_test_actual = np.expm1(y_test)

mae = mean_absolute_error(y_test_actual, y_pred)
rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred))
r2 = r2_score(y_test_actual, y_pred)

print(f"\n📈 Final Model Performance:")
print(f"Mean Absolute Error (MAE): ₹{mae:,.2f}")
print(f"Root Mean Squared Error (RMSE): ₹{rmse:,.2f}")
print(f"R² Score: {r2:.3f}")

# -----------------------------
# 🔟 KFold Cross Validation
# -----------------------------
cv = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(final_pipeline, X, y, cv=cv, scoring='r2')
print(f"\n📊 Cross-Validated R²: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

import joblib

# Save the trained pipeline
joblib.dump(final_pipeline, "models/house_price_pipeline.pkl")
print("✅ Model saved to models/house_price_pipeline.pkl")