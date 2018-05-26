# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 18:38:19 2018

############Excercise Sheet 2 #####################
#SPL SS 18
@author: sydma
"""

#%reset\

import numpy as np
import pandas as pd
import os


###############################################################################
# 1.  Please create a dataframe with the data below, and solve the problems after the table
#  a. Compute the minimum and maximum of each column, print the values and their corresponding country. 
#  b. Compute the range range = maxx−minx for each column
##############################################################################
os.chdir('C:\\Users\\sydma\\Dropbox\\Uni Sach\\Master\\SoSe_18\\Statistical Programming Languages\\02')
os.getcwd()
# Copy paste into Excel then saved as a csv
countries = pd.read_csv('countries.csv')
countries = pd.read_csv('countries.csv',index_col=0)

#a)

#Values
countries['increaseoftheindex(x)'].max()
countries['unemployment(y)'].max()
countries['increaseoftheindex(x)'].min()
countries['unemployment(y)'].min()

#Corresponding countries
countries['increaseoftheindex(x)'].argmax()
countries['unemployment(y)'].argmax()
countries['increaseoftheindex(x)'].argmin()
countries['unemployment(y)'].argmin()

#b)
#Ranges
countries.max()-countries.min()

###############################################################################
# 2 Load the R dataset mtcars and ﬁgure out the variables using help().
##############################################################################
### rpy2 is a neat way to inderact with python
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri

pandas2ri.activate()
R = ro.r
mtcars=pd.DataFrame(R['mtcars'])
pandas2ri.deactivate()


# mtcars=pd.read_table('mtcars.txt',sep=';')
#note use . as decimal makes life easier
mtcars.info()
######### Exercise 3
mtcars.columns
mtcars=mtcars.sort_values(by=['mpg','cyl'],ascending=False)

######### Exercise 4
mtcars.drop(['carb'],axis=1)

######### Test Write Data to SQL Table
from sqlalchemy import create_engine
engine = create_engine('sqlite:///SPL_Test.db')
mtcars.to_sql('TBL_mtcars',engine)

######### Exercise 5
mtcars.columns
r_cars=mtcars[['hp', 'cyl', 'disp', 'mpg', 'drat', 'wt', 'qsec', 'vs', 'am', 'gear',
       'carb']]

######### Exercise 6
########### Test Read Table from SQL
##However had to change Column name
pd.read_sql('SELECT * FROM TBL_mtcars where instr(cars,"Merc")!=0',engine)


mtcars.loc[mtcars.index.str.contains('Merc')]

######### Exercise 7


dax30 = pd.read_csv('dax_prices.csv',parse_dates=[0],)
######### Exercise 8
dax30.info()
dax30.head()
######### Exercise 9
dax30 = dax30.rename(index=str,columns = {' DAX':'DAX Prices'})
######### Exercise 10
dax30.to_csv('dax_prices.txt',sep=';',decimal = ',')
######### Exercise 11
pd.read_csv('dax_prices.txt',sep=';',decimal=',')








