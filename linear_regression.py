import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
df=pd.read_csv("uber.csv")
print(df.head())
#collection x et y 
x=df['Open'].values
y=df['Close'].values
mean_x=np.mean(x)
mean_y=np.mean(y)
n=len(x)
num=0
denum=0
for i in range(n):
    num += (x[i] - mean_x)*(y[i]-mean_y)
    denum +=(x[i] - mean_x)**2
b_1 = num/denum 
b_0 = mean_y - (b_1*mean_x)
print(b_1,b_0)
max_x=np.max(x)+100
min_x=np.min(x)-100
a=np.linspace(min_x,max_x,1000)
b=b_0+b_1*a 
plt.plot(a,b, color='red',label='regression line')
plt.scatter(x,y, color='blue', label='scatter plot')
plt.xlabel('head size in cm3')
plt.ylabel('close day')
plt.legend()
plt.show()
#find if our figure is good 
ss_t=0
ss_r=0
for i in range(n):
    y_pred=b_0+b_1 *x[i]
    ss_t += (y[i]-mean_y) ** 2
    ss_r += (y[i] -y_pred) **2
r_2 = 1- (ss_r/ss_t)
print(r_2)
#methode 2 using sklearn 
x=x.reshape((n,1))
reg=LinearRegression()
reg=reg.fit(x,y)
y_pred=reg.predict(x)
r_2_score= reg.score(x,y)
print(r_2_score)