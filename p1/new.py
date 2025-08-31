import pandas as pd
import numpy as np
from sklearn.linear_model import  LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error

data=pd.read_csv("Untitled - Sheet 1 (2).csv")
X=data[['Hours']]
y=data['score']
model =LinearRegression()
model.fit(X,y)
pre= model.predict(X)

# Evaluate
mae=mean_absolute_error(y,pre)
mse=mean_squared_error(y,pre)
rmse=np.sqrt(mse)

print("MAE",mae)
print("MSE",mse)
print("rmse",rmse)


new_hour=float(input("Enter no. of hours : "))
new_pre=model.predict([[new_hour]])
print(f"Prediction for {new_hour} hours is {new_pre}")