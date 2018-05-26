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

os.chdir('C:\\Users\\sydma\\Dropbox\\Uni Sach\\Master\\SoSe_18\\Statistical Programming Languages\\03')
os.getcwd()

#Exercise 1 Quick Time Series

#a
dax30 = pd.read_table('dax30.txt',decimal=',',index_col = 0)

#b
dax30.plot()
#c
ret = pd.Series(np.diff(np.log(dax30['Index'])),index=dax30.index[1:])
ret.plot()

#d QQ Plots
import pylab 
import scipy.stats as stats
   
stats.probplot(ret, dist="norm", plot=pylab)
pylab.show()


#e Histogram
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

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

###### Exercise 2 

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