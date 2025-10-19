import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
# Load dataset
titanic_df = pd.read_csv('data/Titanic-Data.csv')

# Fill missing values
titanic_df['Age'] = titanic_df['Age'].fillna(titanic_df['Age'].median())
titanic_df['Cabin'] = titanic_df['Cabin'].fillna('Unknown')
titanic_df['Embarked'] = titanic_df['Embarked'].fillna(titanic_df['Embarked'].mode()[0])

# Drop non-numeric or irrelevant columns
titanic_df.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1, inplace=True)

# Encode categorical variables
le=LabelEncoder()
titanic_df['Sex'] = le.fit_transform(titanic_df['Sex'])
titanic_df['Embarked'] = le.fit_transform(titanic_df['Embarked'])

# Define features and target
X = titanic_df.drop('Survived', axis=1)
Y = titanic_df['Survived']

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
# Train model
model = XGBClassifier(n_estimators=150, learning_rate=0.05, max_depth=3, eval_metric='logloss', random_state=42)
model.fit(X_train, Y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("Accuracy score is", accuracy_score(Y_test, y_pred))
print("Classification report:\n", classification_report(Y_test, y_pred))

joblib.dump(model,"models/titanic_survival_model.joblib")