import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report
iris_df=pd.read_csv('data/IRIS.csv')
iris_df.drop_duplicates(inplace=True)
iris_df['sepal_area']=iris_df['sepal_length']*iris_df['sepal_width']
iris_df['petal_area']=iris_df['petal_length']*iris_df['petal_width']
iris_df.drop(columns=['sepal_length','sepal_width','petal_length','petal_width'],inplace=True)
X=iris_df.drop('species',axis=1)
Y=iris_df['species']
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=42)
model=LogisticRegression()
model.fit(X_train,Y_train)
y_pred=model.predict(X_test)
# Accuracy
accuracy = accuracy_score(Y_test, y_pred)
print(f"✅ Model Accuracy: {accuracy:.2f}")

# Classification report
print("\n📊 Classification Report:")
print(classification_report(Y_test, y_pred))
