import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
student_score_df=pd.read_csv('data/student_exam_scores.csv')
student_score_df.drop(columns=['student_id'],axis=1,inplace=True)
import joblib
X=student_score_df.drop('exam_score',axis=1)
Y=student_score_df['exam_score']

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)

model=LinearRegression()
model.fit(X_train,Y_train)
y_pred=model.predict(X_test)

print("R2 Score is :",r2_score(Y_test,y_pred))
print("Mean Absolute error Score is :",mean_absolute_error(Y_test,y_pred))
print("Mean Squared error Score is :",mean_squared_error(Y_test,y_pred))

joblib.dump(model,"models/student_score_predict_model.joblib")


