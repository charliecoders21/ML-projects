import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from imblearn.over_sampling import SMOTE
import joblib
bank_note_auth_df=pd.read_csv('data/BankNote_Authentication.csv')
print( bank_note_auth_df['class'].value_counts())
bank_note_auth_df.drop_duplicates(inplace=True)

X=bank_note_auth_df.drop('class',axis=1)
Y=bank_note_auth_df['class']



X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=42)
smote=SMOTE(random_state=42)
X_train_resample,Y_train_resample=smote.fit_resample(X_train,Y_train)
model=LogisticRegression(max_iter=1000,random_state=42)
model.fit(X_train_resample,Y_train_resample)
y_model_pred=model.predict(X_test)
mae=mean_absolute_error(Y_test,y_model_pred)
mse=mean_squared_error(Y_test,y_model_pred)
r2=r2_score(Y_test,y_model_pred)
print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")
joblib.dump(model,'best_model.joblib')




