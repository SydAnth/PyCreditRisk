# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 10:28:12 2018
########## Exercise Sheet 4 #########

@author: sydma
"""

import numpy as np
import pandas as pd
from functools import reduce
import os
from scipy import integrate
os.chdir('C:\\Users\\sydma\\Dropbox\\Uni Sach\\Master\\SoSe_18\\Statistical Programming Languages\\04')
os.getcwd()


###############################################################################
# 1. Generate 100 random samples (x) from a normal distribution with µ = 0, σ = 2. 
#    For each element of x compute y, such that y1 = 0 and 
#
##############################################################################

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
        
###############################################################################
# 2. Compute max and min of the variable Sepal.Length for each species of the iris data. 
#(Hint: use the family of apply() functions)
#
##############################################################################

iris = pd.read_table('iris.txt',sep=';',decimal='.')

iris.columns

iris.groupby(['Species',]).max()['Sepal.Length']
iris.groupby(['Species',]).min()['Sepal.Length']
##############################################################################
# Excursion: How to index Pandas Dataframe 
#
##############################################################################

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

###############################################################################
# 3. Write a function for deleting rows of a data frame containing NAs (missing values).
#     (Hint: use function notnull() method
#
##############################################################################
# pd.notnull() returns a Dataframe with Booleans .any() reduces this to a 
# series with which to index the original dataframe

def clean_data_frame(inpt_frame):

     out_frame = inpt_frame[inpt_frame.notnull().any(axis=1)]
     return out_frame 

clean_data_frame(dax30)

##### Existing Function in pandas library
dax30.dropna()

###############################################################################
#4. Write a recursive function, which returns the n Fibonaci number 
#   y_n = y_n−1 + y_n−2,∀n > 2. (Hint: do not use loops and think about the ﬁrst numbers)
##############################################################################

### Takes way too long for large numbers
def recursive_fibonacci(n):
    if n <= 1:
        return n
    else: 
        return recursive_fibonacci(n-1) + recursive_fibonacci(n-2)


recursive_fibonacci(10)

#### Alternative Solution
#### Works with large numbers but is a cheat since reduce basically
#### replaces for loop
fib = lambda n:reduce(lambda x,n:[x[1],x[0]+x[1]], range(n),[0,1])[0]
fib(10)       

###############################################################################
# 5. Write functions, to compute the densities and probabilities for a 
#    random variable following the normal distribution X ∼ N(µ,σ2)
#
##############################################################################

###Density
# Note: ** are power operations in Python

def normal_density(x,mu=0,sigma=1):   
        return(1/(2*np.pi*sigma**2)**0.5*np.exp(-(x-mu)**2/(2*sigma**2)))
        
###Normal Distribution        
def cdf_normal(x):
    return integrate.quad(lambda x: normal_density(x),-np.inf,x)[0]

normal_density(5)
cdf_normal(5)


###############################################################################
# 6. Write a function to compute OLS coeﬃcients and standard errors for a dataframe, 
#    take into account the possibility of missing values. 
#    (Hint: Use the dataset EuStockMarkets as a starting point.)
#
##############################################################################

EUStoxx=pd.read_table('EuStockMarkets.txt',sep=';',decimal='.')


def OLS_regression(input_frame,y,X,constant = True):
    import pandas as pd
    import numpy  as np
    
    y=input_frame[y].dropna() #Drop Misings from Dataframe

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


###############################################################################
# For purposes of Comparison 2 popular Python Libraries for Linear Regression
# models are introduced
##############################################################################

# With Statsmodels
import statsmodels.api as sm 

y = EUStoxx['DAX']
X = sm.add_constant(EUStoxx[[x for x in EUStoxx.columns if x!= 'DAX']])
lm = sm.OLS(y,X).fit()
lm.summary()
## Way better Output


# With SKlearn
#No need to add constant to dataset since constant
#default setting of function
X =EUStoxx[[x for x in EUStoxx.columns if x!= 'DAX']]
y = EUStoxx['DAX']
from sklearn import linear_model
lm2 = linear_model.LinearRegression()
lm2.fit(X,y)
print(lm2.intercept_,lm2.coef_)
