# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 10:28:12 2018
########## Exercise Sheet 4 #########

@author: sydma
"""

import numpy as np
import pandas as pd
import scipy
import datetime as dte
import os

os.chdir('C:\\Users\\sydma\\Dropbox\\Uni Sach\\Master\\SoSe_18\\Statistical Programming Languages\\04')
os.getcwd()

###Exercise 1


np.random.seed(1)
x = np.random.normal(0,2,100)
y = np.full((100,),0).astype(dtype='float')

for i in range(0,len(x)-1):
    if x[i]>=0:
        y[i+1]=y[i]+x[i]        
    elif -1<x[i] and x[i]<0:
        y[i+1]=y[i]-2*x[i]
    else:     
        y[i+1]=y[i]-x[i]
        
"""
###### Excercise 2
"""
iris = pd.read_table('iris.txt',sep=';',decimal='.')

iris.columns

iris.groupby(['Species',]).max()['Sepal.Length']
iris.groupby(['Species',]).min()['Sepal.Length']


###### Excercise 3
dax30 = pd.read_csv('dax_prices.csv',parse_dates=[0],)
## Rename because after import column name ' DAX'
dax30 = dax30.rename(index=str,columns = {' DAX':'DAX'})
###Pandas Indexing Excursion
###Using Integers for Location
dax30.iloc[[1],[1]]
###Using Indexes
dax30.loc[['1'],['DAX']]
###Using Boolean Logic
dax30[dax30['DAX']>7000]
### Combination of Boolean logic
### Parenthesis are necessary to combine statements
dax30[(dax30['DAX']>7000) & (dax30['Date'].dt.year == 2012)]

###### Indexing NaNs
# pd.notnull() returns a Dataframe with Booleans .any() reduces this to a 
# series with which to index the original dataframe

def clean_data_frame(inpt_frame):

     out_frame = inpt_frame[inpt_frame.notnull().any(axis=1)]
     return out_frame 

clean_data_frame(dax30)

##### Existing Function in pandas library
dax30.dropna()

######## Exercise 4 Write a function which returns Fibonacci Number

### Takes way too long for large numbers
def recursive_fibonacci(n):
    if n <= 1:
        return n
    else: 
        return recursive_fibonacci(n-1) + recursive_fibonacci(n-2)


recursive_fibonacci(10)

#### Works with large numbers

def quicker_fibonacci(n):
    x2 = 0
    x1 = 1
    for i in range(0,n-1):
         x = x1 + x2
         x2 = x1
         x1 = x

    return x
     
quicker_fibonacci(10)       

######## Exercise 5 Write a function which computes densities and cdf

###Density
# Note: ** are power operations in Python


"""
class random_var:
    
        def __init__(self,name,x,distribution,mu,sigma):
                self.name = name
                self.x = x
                self.distribution = distribution
                self.mu = mu
                self.sigma = sigma
        
        def normal_density(x,mu=0,sigma=1):
            import numpy as np
            return(1/(2*np.pi*sigma**2)**0.5*np.exp(-(x-mu)**2/(2*sigma**2)))
        
        def cdf_normal(x):
            import scipy
            return scipy.integrate.quad(lambda x: normal_density(x),-np.inf,x)[0]


random_var.name
"""


def normal_density(x,mu=0,sigma=1):
        import numpy as np
        return(1/(2*np.pi*sigma**2)**0.5*np.exp(-(x-mu)**2/(2*sigma**2)))
        
def cdf_normal(x):
    from scipy import integrate
    return scipy.integrate.quad(lambda x: normal_density(x),-np.inf,x)[0]


normal_density(5)
cdf_normal(5)
#Calc Check
from scipy.stats import norm
norm.pdf(5)
norm.cdf(5)

######## Exercise 6 Write a function which calculates OLS Coefficients

EUStoxx=pd.read_table('EuStockMarkets.txt',sep=';',decimal='.')

#select = [col for col in input_frame.columns if col != '%s'%(y)]

def OLS_regression(input_frame,y,X,constant = True):
    import pandas as pd
    import numpy  as np
    
    y=input_frame[y]

    if constant == True:
        X = input_frame[[col for col in X]]
        pd.options.mode.chained_assignment = None  # default='warn'
        X['Constant'] = pd.Series(np.full((len(X),),1),index=np.arange(1,len(X)+1,1))
        pd.options.mode.chained_assignment = 'warn'  # default='warn'
    else:
        X = input_frame[[col for col in X]]
        
    beta_hat = np.matmul(np.matmul(np.linalg.inv(np.matmul(X.transpose(),X)),X.transpose()),y)
    res       = pd.Series(y - np.matmul(X,beta_hat),name = 'Residuals')
    SSE       = np.matmul(res.transpose(),res)/(len(res)-len(beta_hat))
    
    Results = pd.Series(beta_hat,index = X.columns,name='Values')
    Results['SSE'] = SSE
    return Results

OLS_regression(EUStoxx,'DAX',['CAC','SMI','FTSE']) 
#### Using Existing Software for Liner Regression Models

# With Statsmodels
import statsmodels.api as sm 

y = EUStoxx['DAX']
X = sm.add_constant(EUStoxx[[x for x in EUStoxx.columns if x!= 'DAX']])
lm = sm.OLS(y,X).fit()
## Way better Output


# With SKlearn
#No need to add constant to dataset since constant
#default setting of function
X =EUStoxx[[x for x in EUStoxx.columns if x!= 'DAX']]
y = EUStoxx['DAX']
from sklearn import linear_model
lm2 = linear_model.LinearRegression()
lm2.fit(X,y).summary()
print(lm2.intercept_,lm2.coef_)