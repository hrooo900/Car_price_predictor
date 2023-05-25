"""
Created on Sat Jul  9 19:13:05 2022

@author: Hritwij
"""

#%%
import pandas as pd
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split


dataset = pd.read_csv('dataset.csv')
dataset = dataset.drop(['car_ID'],axis=1)

# Summarizing Dataset

print(dataset.shape)
print(dataset.head(5))

#%%

Xdata = dataset.drop('price',axis='columns')

numericalCols=Xdata.select_dtypes(exclude=['object']).columns

X=Xdata[numericalCols]

Y = dataset['price']

cols = X.columns

X = pd.DataFrame(scale(X))
X.columns = cols


#%%

x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.20,random_state=0)

from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(x_train, y_train)

#%%

ypred = model.predict(x_test)

from sklearn.metrics import r2_score
r2score = r2_score(y_test,ypred)
print("R2Score",r2score*100)