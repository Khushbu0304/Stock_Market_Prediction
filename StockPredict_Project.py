import numpy as np
import pandas as pd
import math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from matplotlib import style

style.use('ggplot')
import matplotlib.pyplot as plt
dataset=pd.read_csv('AppleStock.csv')
dataset.head()

dataset.drop('Adj Close',axis=1,inplace=True)
dataset.head()

dataset.plot(kind='line',subplots=True,layout=(1,5),sharex=False,sharey=False)
plt.show()

dataset.hist()
plt.show()

dataset['OC_Change']=(dataset['Close']-dataset['Open'])/dataset['Open']*100
dataset['HL_Change']=(dataset['High']-dataset['Low'])/dataset['Low']*100
print(dataset.tail())
forecast_out=int(math.ceil(0.1*len(dataset)))
dataset['Future_Price']=dataset['Close'].shift(-forecast_out)
print(dataset['Future_Price'])
print(dataset.head())
storedata=dataset[-forecast_out:]
dataset=dataset.dropna()
print(dataset.tail())
print(dataset.count())
dataset=dataset[['Date','Close','HL_Change','OC_Change','Volume','Future_Price']]
print(dataset.tail())
X=np.array(dataset.drop(['Future_Price'],axis=1))
y=np.array(dataset['Future_Price'])
#print(len(X),len(y))
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
date_train=np.array(X_train[:,0])
X_train=np.array(X_train[:,1:], dtype=np.float)
X_test=np.array(X_test[:,1:], dtype=np.float)
linearRegressor=LinearRegression()
linearRegressor.fit(X_train,y_train)


linearAccuracy=linearRegressor.score(X_test,y_test)
print("linear accuracy ",linearAccuracy);
print(forecast_out)

#X_old=X[:-forecast_out]
X_new=X[-forecast_out:]
date_new=np.array(storedata['Date'])
X_new=np.array(X_new[:,1:], dtype=np.float)
predictedValue=linearRegressor.predict(X_new)
date_oct=[]
predict_oct=[]
date_nov=[]
predict_nov=[]
date_sept=[]
predict_sept=[]

for i in range(len(predictedValue)):
    print("On",date_new[i],"predicted value",predictedValue[i])
    
#plot for whole data
plt.plot(predictedValue)
plt.title("Stock price prediction (Testing Set)")
plt.xlabel("Trend Prediction towards the next %d"%len(predictedValue)+" days")
plt.ylabel("closing price ")
plt.xticks(rotation=90)
plt.show()    

for i in range(len(predictedValue)):
    j=date_new[i].index('-')
    string=date_new[i][j+1]+date_new[i][j+2]
    if(string=="09"):
        date_sept.append(date_new[i])
        predict_sept.append(predictedValue[i])
    if(string=="10"):
        date_oct.append(date_new[i])
        predict_oct.append(predictedValue[i])
    if(string=="11"):
        date_nov.append(date_new[i])
        predict_nov.append(predictedValue[i])

#plot for month of september
plt.plot(date_sept,predict_sept)
plt.title("Stock price prediction for month of September")
plt.xticks(date_sept,date_sept,rotation="vertical")
plt.ylabel("closing price ")
plt.show()

#plot for month of october
plt.plot(date_oct,predict_oct)
plt.title("Stock price prediction for month of October")
plt.xticks(date_oct,date_oct,rotation="vertical")
plt.ylabel("closing price ")
plt.show()

#plot for month of november
plt.plot(date_nov,predict_nov)
plt.title("Stock price prediction for month of November")
plt.xticks(date_nov,date_nov,rotation="vertical")
plt.ylabel("closing price ")
plt.show()
#print(X_new)
