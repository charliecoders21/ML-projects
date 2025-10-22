import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.pipeline import Pipeline
import joblib
bank_note_auth_df=pd.read_csv('data/BankNote_Authentication.csv')
bank_note_auth_df.drop_duplicates(inplace=True)

X=bank_note_auth_df.drop('class',axis=1)
Y=bank_note_auth_df['class']



X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=42)

models={'Ridge':Ridge(),'Lasso':Lasso(),"ElasticNet":ElasticNet()
}
params={
    'Ridge':{'model__alpha':[0.01,0.1,1,10,100]},
    'Lasso':{'model__alpha':[0.001,0.01,0.1,1,10]},
    'ElasticNet':{'model__alpha':[0.01,0.1],'model__l1_ratio':[0.1,0.5,0.9]}
}

best_model={}
for name,model in models.items():
    pipeline=Pipeline([
        ('poly',PolynomialFeatures(degree=2,include_bias=False)),
        ('scaler',StandardScaler()),
        ('model',model)
    ])
    grid=GridSearchCV(pipeline,params[name],cv=5,scoring='r2',n_jobs=-1)
    grid.fit(X_train,Y_train)
    y_model_pred=grid.predict(X_test)
    mae=mean_absolute_error(Y_test,y_model_pred)
    mse=mean_squared_error(Y_test,y_model_pred)
    r2=r2_score(Y_test,y_model_pred)
    print(f"Mean Absolute Error: {mae}")
    print(f"Mean Squared Error: {mse}")
    print(f"R^2 Score: {r2}")
    best_model[name]=grid.best_estimator_
    print(f"Best parameters for {name}: {grid.best_params_}")
    
best_model_name=max(best_model,key=lambda x:cross_val_score(best_model[x],X,Y,cv=10,scoring='r2').mean())
joblib.dump(best_model[best_model_name],'best_model.joblib')
    







