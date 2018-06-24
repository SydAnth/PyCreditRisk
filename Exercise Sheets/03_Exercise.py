# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 15:01:38 2018
############Excercise Sheet 3 #####################
#SPL SS 18
@author: sydma
"""
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import pylab 
import scipy.stats as stats 

os.chdir('C:\\Users\\sydma\\Dropbox\\Uni Sach\\Master\\SoSe_18\\Statistical Programming Languages\\03')
os.getcwd()

###############################################################################
#   1. Using the ﬁle dax30.txt create:
#       (a) Read in the text ﬁle dax30.txt. 
#       (b) Plot the time series for the DAX index from the dax30.txt.
#       (c) Plot the daily log returns of the DAX index.
#       (d) Compare the sample quantiles from the DAX log-returns with 
#           the theoretical quantiles from a normal distribution using QQ-plots.
#       (d1) Try to do it in general for t-distribution
#       (e) Create a histogram for the DAX log returns with 44 breaks.
#           Furthermore add estimated densities using the following Kernels:
#           epanechnikov and gaussian with a bandwidth of 0.01.
#       (f) Save the plots in .pdf and .png format.
##############################################################################

#a
dax30 = pd.read_table('dax30.txt',decimal=',',index_col = 0)
#b
dax30.plot()
#c
ret = pd.Series(np.diff(np.log(dax30['Index'])),index=dax30.index[1:])
ret.plot()

#d QQ Plots
  
stats.probplot(ret, dist="norm", plot=pylab)
pylab.show()


fig = plt.figure()
ax = fig.add_subplot(111)
stats.probplot(ret, dist=stats.t ,sparams=(len(ret)-1,) , plot = ax)
ax.set_title("Probplot for T dist with n -1 Degrees of Freedom")
plt.show()


#e Histogram


plt.hist(ret,bins=44,normed=True)
plt.xlim((-0.1,0.1))
mean = np.mean(ret)
variance = np.var(ret)
sigma = np.sqrt(variance)
x = np.linspace(min(ret), max(ret), 100)
plt.title('DAX Daily Log Returns')
plt.plot(x, mlab.normpdf(x, mean, sigma))
plt.savefig(fname='SPL_03_01_f.png') # 1 f
plt.show()



###############################################################################
# 2. (HW after the course ;) ): Use the dataset USPersonalExpenditure to create 
#     an area plot. Use loops. Eventually program it as a general function.
##############################################################################

USExp=pd.read_csv('PersExp.txt',sep=',')

RelExp = (USExp / USExp.sum()).transpose()
df = RelExp.cumsum(axis=1)

colors = ['red','green','blue','black','brown'] 
fig, ax = plt.subplots()
for i in range(0,len(df.columns)):
    ax.plot(df.index,df.iloc[:,i], color=colors[i],
            label=df.columns[i],alpha=1)
    if i == 0:
        ax.fill_between(RelExp.index, 0, df.iloc[:,i]) 
    else:
        ax.fill_between(RelExp.index,  df.iloc[:,i-1], df.iloc[:,i]) 
        
ax.set_title('Relative Share of US Expenditures')
ax.legend(loc='lower left')
plt.show()


###### Put it in a fct #########

def Area_Plot(df,graph_name):
    
    df = (df / df.sum()).transpose()
    df = df.cumsum(axis=1)
    
    colors = ['red','green','blue','black','brown'] 
    fig, ax = plt.subplots()
    for i in range(0,len(df.columns)):
        ax.plot(df.index,df.iloc[:,i], color=colors[i],
                label=df.columns[i],alpha=1)
        if i == 0:
            ax.fill_between(RelExp.index, 0, df.iloc[:,i]) 
        else:
            ax.fill_between(RelExp.index,  df.iloc[:,i-1], df.iloc[:,i]) 
            
    ax.set_title(graph_name)
    ax.legend(loc='lower left')
    plt.show()
    
    
    return

##### Calling that fct ########
    
Area_Plot(USExp,'US Expenditures')