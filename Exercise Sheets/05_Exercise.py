# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 18:02:03 2018
########## Exercise Sheet 5 #########
@author: sydma
"""

import numpy as np
import pandas as pd
import scipy.stats as st
import scipy
import os

os.chdir('C:\\Users\\sydma\\Dropbox\\Uni Sach\\Master\\SoSe_18\\Statistical Programming Languages\\05')
os.getcwd()

########Part 1#############################

x = np.array([2.8, 1.2, 2.1, 1.6, 1.5, 4.6, 3.6, 2.1, 6.5, 4.6, 3.0, 1.3, 4.2])
y = np.array([9.4, 10.4, 10.8, 10.5, 18.4, 11.1, 2.6, 8.8, 5.0, 21.5, 6.7, 2.5, 5.6])
countries = np.array(["Belgium", "Denmark", "France", "GB", "Ireland", "Italy", "Luxembourg", "Holland", "Portugal", "Spain", "USA", "Japan", "Deutschland"])


myframe = pd.DataFrame(np.column_stack((x,y)),index=countries,columns=['incr_indx','unemp'])

# Exercise 1: max, min (and corresponding country)

print(myframe['incr_indx'].argmin(),myframe['incr_indx'].min())
print(myframe['incr_indx'].argmax(),myframe['incr_indx'].max())
print(myframe['unemp'].argmin(),myframe['unemp'].min())
print(myframe['unemp'].argmax(),myframe['unemp'].max())

# Exercise 2: range = maxx−minx
# ptp is the pyhton equivalent to range fct
myframe['unemp'].ptp()
myframe['incr_indx'].ptp()

#Exercise 3: Quantiles + #Exercise 4 Median
#Tuple that shit

myframe['incr_indx'].quantile((.25,.5,.75))
myframe['unemp'].quantile((.25,.5,.75))
#short cut
myframe.quantile((.75,.5,.25))

#Exercise 5: Interquantile Differences
np.diff(myframe['incr_indx'].quantile((.25,.75)))
np.diff(myframe['unemp'].quantile((.25,.75)))

#Exercise 6: Means
myframe.mean()

#Excercise 7:MAD
myframe['incr_indx'].mad()
myframe['unemp'].mad()

#Exercise 8: Variance

myframe.var()

#Exercise 9: Standart Deviation
myframe.var()**.5

#Exercise 10: Covariance
myframe.cov()

#Exercise 11: Correlation
myframe.corr()

#Exercise 12: Ranks
myframe.rank()

#Exercise 13: Rank Correlation
myframe.corr(method='spearman')

#Exercise 14: Confidence Interval



#Get Critical Values
z_value = st.norm.ppf((.025,.975))
t_value = st.t.ppf((.025,.975),df=12)

def Confidence_Int(series,var_known=False):
    
        if var_known == True:
            conf = series.mean() + z_value * series.var()**.5/len(series)**.5
        else:
            conf = series.mean() + t_value * series.var()**.5/len(series)**.5
       
        return conf

Confidence_Int(myframe['unemp'],True)


########Part 2#############################

#Exercise 1

# Generate data
eps=np.random.normal(0,1,100)
X = np.random.uniform(low=-1,high=1,size=100)
Y = X**2 + .5*eps

# Covariance between X, Y
np.cov(X,Y)
# Make a plot
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.scatter(X, Y,alpha=0.5)
ax.set_xlabel('X', fontsize=15)
ax.set_ylabel('Y', fontsize=15)
ax.set_title('Plot X on Y')
ax.grid(True)
fig.tight_layout()
plt.show()
#summary of Y
st.describe(Y)
#Liner Reg Model
import statsmodels.api as sm 

lm = sm.OLS(Y,X).fit()
#Coefficients From LM
lm.params
#Standard Errors (Coefficients) From LM
lm.bse
#LM Fitted Values
lm.fittedvalues
#LM Residuals
lm.resid

### Plottting Density in Python 
#Step 1: Determine KDE
#Recall: kernel density estimation (KDE) is a non-parametric way to 
#estimate the probability density function of a random variable.
kde  = st.gaussian_kde(lm.resid,bw_method=0.2)
#Set up Linear Space (from min to max of the random var) #Number of samples
#This is the x-axis for the graph
x_grid = np.linspace(lm.resid.min(), lm.resid.max(), 200)
#estimated pdf
pdf = kde.evaluate(x_grid)
#True Ppdf
pdf_true=st.norm.pdf(x_grid)
#Draw Graph

fig, ax = plt.subplots()
ax.plot(x_grid,pdf, color='blue',label='pdf_kde', alpha=0.5, lw=3)
ax.plot(x_grid,pdf_true, color='red',label='pdf_norm', alpha=0.5, lw=3,)
ax.set_title('Denisty of residuals')
ax.legend(loc='upper left')
plt.show()

#Calc Empirical Cumulative Distribituin Function  
def ecdf(x):
    xs = np.sort(x)
    ys = np.arange(1, len(xs)+1)/float(len(xs))
    return xs, ys

#Calc Empirical Cumulative Distribution Function  
#Alongside CDF of normal distribution
"This needs to be checked"
empCDF=ecdf(lm.resid)
cdf_norm=st.norm.cdf(x_grid)
fig, ax = plt.subplots()
ax.plot(empCDF[0],empCDF[1], color='blue',label='CDF_emp', alpha=0.5, lw=3)
ax.plot(x_grid,cdf_norm, color='red',label='CDF_norm', alpha=0.5, lw=3,)
ax.set_title('CDF of residuals')
ax.legend(loc='upper left')
plt.show()

#QQ-Plot of residuals
#Normal distribution as reference
import pylab 

st.probplot(lm.resid, dist="norm", plot=pylab)
pylab.show()


#Exercise 2 Hubble

hubble = pd.read_table('hubble.txt',sep=' ')
hubble.columns

fit = sm.OLS(hubble['rec.vel'],hubble['distance']).fit()
fit.summary()

### Plot Fitted Values(line) against actual values(Dots)
fig, ax = plt.subplots()
ax.plot(hubble['distance'],fit.fittedvalues, color='blue',label='fitted', alpha=0.5, lw=2)
ax.scatter(hubble['distance'],hubble['rec.vel'], color='red',label='obs', alpha=0.5)
ax.set_title('Regression Velocity on Distance')
ax.legend(loc='upper left')
plt.show()



#Exercise 3 Cereal

cereal=pd.read_table('cereal.txt')

select = [x for x in cereal.columns if x != 'rating']

model = sm.OLS(cereal['rating'],cereal[select]).fit()

model.summary()

### module necessary for 3D-Plots
from mpl_toolkits.mplot3d import Axes3D

#Creating a 3D-Plot for Linear Regression Model 

#Step 1: Numpy Meshgrid function to create Regression Plane 
x_surf,y_surf = np.meshgrid(np.linspace(cereal['fat'].min(), cereal['fat'].max(), 10),
                            np.linspace(cereal['sugars'].min(), cereal['sugars'].max(), 10))


#Step 2: Numpy Meshgrid function to create Plane for fitted Values
onlyX = pd.DataFrame({'fat': x_surf.ravel(), 'sugars': y_surf.ravel()})
fittedY= model.predict(exog=onlyX)
#Note ravel() turns matrix into verctor

#Step 3: Plot Graph
fig = plt.figure() 
ax = fig.add_subplot(111,projection='3d')

ax.scatter3D(cereal['fat'],cereal['sugars'],cereal['rating'], c='red',alpha=.5,label='rating')
ax.plot_surface(x_surf,y_surf,fittedY.reshape(x_surf.shape),rstride=1,
                cstride=1,
                color='None',
                alpha = 0.4)

ax.set_title('Regression of rating on fat and sugars')
ax.legend(loc='lower left')
ax.set_xlabel('fat')
ax.set_ylabel('sugars')
ax.set_zlabel('rating')
plt.show()

help(ax.plot_surface)


"""
#Exercise 4
Generate 200 samples from an ARMA(2,1) model with 
µ = 0 and (a1,a2,b1) = (0.6,−0.8,0.75)
"""
import statsmodels.api as sm 
import statsmodels.tsa.stattools as stools
import statsmodels.graphics.tsaplots as tplot
#Step 1 Generate Arma Process
np.random.seed(100)

arparams = np.array([0.6,-0.8])
maparams = np.array([.75])

ar = np.r_[1, -arparams] # add zero-lag and negate
ma = np.r_[1, maparams] # add zero-lag
y = sm.tsa.arma_generate_sample(ar, ma, 100)

#model = sm.tsa.ARMA(y, (2, 1)).fit(trend='nc', disp=0)

#Descriptive TSA Statistics
stools.adfuller(y)
stools.kpss(y)
#Plot ACF and PACF
tplot.plot_acf(y)
tplot.plot_pacf(y)

#Fir ARMA Model
tsmodel = sm.tsa.ARMA(y, (2, 1)).fit(trend='nc', disp=0)

residuals = tsmodel.resid

stools.q_stat(tsmodel.resid,nobs=len(tsmodel.resid))

fig = plt.figure() 
qq_ax = fig.add_subplot()
sm.qqplot(y, line='s', ax=qq_ax)
qq_ax.set_title('QQ Plot')  #Why no work ?
plt.show()


"""
#Exercise 5 Faithful
Load package datasets. Use data(faithful) to import the waiting time 
(in min) between eruptions and the duration (in min) of the eruption 
for the Old Faithful geyser in Yellowstone National Park, Wyoming, USA. 
We would like to forecast when the next ejection would be.
"""

faithful = pd.read_csv('faithful.txt',sep=',')

#This is Bad why aren't the results the same 
#as in ADF-Test in R Test Statistic about -6.532
adf = stools.adfuller(faithful.waiting)[0]
print('ADF-Test Statistic: %s' %adf)

## Lag-Order from plots ARMA(2,0)
tplot.plot_acf(faithful['waiting'],lags=20)
tplot.plot_pacf(faithful['waiting'],lags=20)

# Can't use Pandas DataFrame if the
#index doesn't have dates
waiting = np.array(faithful['waiting'])

tsmodel = sm.tsa.ARMA(waiting, (2, 0)).fit(trend='nc', disp=0)

fig, ax = plt.subplots()
ax.plot(faithful.index,tsmodel.resid, color='blue',label='Residuals', alpha=0.5, lw=3)
ax.set_title('Residuals of Waiting TIme')
ax.legend(loc='upper left')
plt.show()

Q,p_value=stools.q_stat(tsmodel.resid,nobs=len(tsmodel.resid))

print('Box-Pierce test \n'
       'X-Squared: %s \n'  
       'p-Value: %s' %(Q[1],p_value[1]))

#Jesus Mary and Josef hopefully this makes sense
tsmodel.predict(len(waiting),len(waiting)+5)



"""
#Exercise 6
Simulate 500 samples from a GARCH(1,1) model with (ω,α1,β1) = (0.3,0.4,0.36), 
ε ∼ N(0,1). Plot the time series. Plot the ACF and PACF 
of the time series and the squared time series. 
Fit GARCH(1,1), GARCH(2,1) and GARCH(1,2) to 
the simulated time series. Compare the ACF, PACF of 
the residuals of the three ﬁttings.
"""

np.random.seed(100)

ω = 0.2
a1 = 0.4
b1 = 0.36

n = 500
w = np.random.normal(size=n)
#zeros_like returns an array of 0s same shape as input
eps = np.zeros_like(w)
sigsq = np.zeros_like(w)

for i in range(1, n):
    sigsq[i] = ω + a1*(eps[i-1]**2) + b1*sigsq[i-1]
    eps[i] = w[i] * np.sqrt(sigsq[i])

#Simple Plot
pd.Series(eps).plot()

#
tplot.plot_acf(eps,lags=20)
tplot.plot_acf(sigsq,lags=20)
tplot.plot_pacf(eps,lags=20)
tplot.plot_pacf(sigsq,lags=20)


from arch import arch_model

help(arch_model)
# Estimate GARCH(1,1) model
# Choose this model since AIC is the lowest
am1 = arch_model(eps,vol='Garch')
res1 = am1.fit(update_freq=5)
print(res1.summary())
res1.aic
# Estimate GARCH(2,1) model
#GARCH Volatility Model is standard setting
am2 = arch_model(eps,p=2,q=1)
res2 = am2.fit(update_freq=5)
res2.aic
# Estimate GARCH(1,2) model
am3 = arch_model(eps,p=1,q=2)
res3 = am3.fit(update_freq=5)
res3.aic

help(arch_model.resids)
def plot_resids(x):
    tplot.plot_acf(x.resid,lags=20)
    tplot.plot_pacf(x.resid,lags=20)
    
plot_resids(res1)
plot_resids(res2)  
plot_resids(res3)


"""
#Exercise 7
Download the S&P500 data from Moodle, 
and ﬁt a proper time series model for that. 
• Plot the time series 
• Take log-diﬀerence, plot it. 
See if there are volatility clustering. 
• Fit proper time series model 
• Use summary() and plot() to check the model(s)
"""


sp500 = pd.read_csv('SP500.csv',index_col=0)
#Sort Descending Date
sp500 = sp500.sort_index()

sp500.Close.plot()
sp500.Volume.plot()

#Calculate Returns
ret = pd.Series(np.diff(np.log(sp500['Close'])),index=sp500.index[1:])
ret.plot()
#### yes there is volatility clustering

##### Determine best Time Series Model
def _get_best_model(TS):
    best_aic = np.inf 
    best_order = None
    best_mdl = None

    pq_rng = range(4) # [0,1,2,3,4]
    for i in pq_rng:
            for j in pq_rng:
                try:
                    tmp_mdl = arch_model(ret,p=i,q=j).fit(update_freq=5)
                    tmp_aic = tmp_mdl.aic
                    if tmp_aic < best_aic:
                        best_aic = tmp_aic
                        best_order = (i,j)
                        best_mdl = tmp_mdl
                except: continue
    print('aic: {:6.5f} | order: {}'.format(best_aic, best_order))                    
    return best_aic, best_order, best_mdl



aic,order,mdl=_get_best_model(ret)

### Plot Model takes a while
mdl.plot()
mdl.summary()