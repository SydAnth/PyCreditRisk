[<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/banner.png" width="888" alt="Visit QuantNet">](http://quantlet.de/)

## [<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/qloqo.png" alt="Visit QuantNet">](http://quantlet.de/) **PyEx2** [<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/QN2.png" width="60" alt="Visit QuantNet 2.0">](http://quantlet.de/)

```yaml

Name of QuantLet : PyEx2

Published in : SPL

Description : Solutions to Exercises Day 2

Keywords : 'data frames, indexing, read data, write data'

Author : Tobias Blücher, Niklas Kudernak, Sydney Richards

```



### Pyhton Code:
```python


import pandas as pd
import os


###############################################################################
# 1.  Please create a dataframe with the data below, and solve the problems after the table
#  a. Compute the minimum and maximum of each column, print the values and their corresponding country. 
#  b. Compute the range range = maxx-minx for each column
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

##############################################################################
# 2 Load the R dataset mtcars and ?gure out the variables using help().
##############################################################################
### rpy2 is a neat way to inderact with r in python
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri

pandas2ri.activate()
R = ro.r
mtcars=pd.DataFrame(R['mtcars'])
pandas2ri.deactivate()


# mtcars=pd.read_table('mtcars.txt',sep=';')
#note use . as decimal makes life easier
mtcars.info()
##############################################################################
# 3 Sort mtcars by columns mpg and cyl in descending order.
##############################################################################
mtcars.columns
mtcars=mtcars.sort_values(by=['mpg','cyl'],ascending=False)

##############################################################################
# 4 Remove the column carb.
##############################################################################
mtcars.drop(['carb'],axis=1)


##############################################################################
# 5 Switch the columns mpg and hp and save the resulting 
#   data frame into a variable r.cars.
##############################################################################
mtcars.columns
r_cars=mtcars[['hp', 'cyl', 'disp', 'mpg', 'drat', 'wt', 'qsec', 'vs', 'am', 'gear',
       'carb']]

##############################################################################
# 6 Extract only the cars of brand Mercedes by ?nding the indices 
#   of row names containing the string ’Merc’ (Hint: use function grep).
##############################################################################

mtcars.loc[mtcars.index.str.contains('Merc')]

##############################################################################
# 7 Read the ?le dax prices.csv into a data frame dax.prices
##############################################################################

dax30 = pd.read_csv('dax_prices.csv',parse_dates=[0],)
##############################################################################
# 8 Inspect the ?rst rows of dax.prices. Are the prices correctly 
#   interpreted as numeric?
##############################################################################

dax30.info()
dax30.head()


##############################################################################
# 9 Rename the column DAX to DAX Prices
#
##############################################################################

dax30 = dax30.rename(index=str,columns = {' DAX':'DAX Prices'})


##############################################################################
# 10 Write dax.prices to disk as .txt ?le, with “;” as separator 
#     and “,” as decimal point.
##############################################################################


dax30.to_csv('dax_prices.txt',sep=';',decimal = ',')


##############################################################################
# 11 Read dax prices.txt into a data frame dax.prices.txt. 
#    Make sure the prices are correctly interpreted as numeric.
##############################################################################
pd.read_csv('dax_prices.txt',sep=';',decimal=',')


```