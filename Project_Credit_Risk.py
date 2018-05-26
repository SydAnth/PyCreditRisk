# -*- coding: utf-8 -*-
"""
Created on Tue May 15 17:57:21 2018

@author: sydma
"""


import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import scipy.stats as st
from sklearn import preprocessing 
from sklearn.metrics import roc_auc_score
import statsmodels.api as sm
import statsmodels.tools as smt



import plotly.plotly as py
import plotly.graph_objs as go
import plotly 


plotly.tools.set_credentials_file(username='SPL2018SS', api_key='X8qHACTMQmxmmsnQNfmD')

############# Import Data ###################

data_path = 'C:\\Users\\sydma\\Dropbox\\Uni Sach\Master\\SoSe_18\\Statistical Programming Languages\\Topic\\Loan_Club\\'
engine = create_engine('sqlite:///' + data_path +'database.sqlite')
df = pd.read_sql('SELECT * FROM loan',engine)

"""
numerical_cols=['sub_grade_num', 'short_emp', 'emp_length_num','dti', 'payment_inc_ratio', 'delinq_2yrs', \
                'delinq_2yrs_zero', 'inq_last_6mths', 'last_delinq_none', 'last_major_derog_none', 'open_acc',\
                'pub_rec', 'pub_rec_zero','revol_util']

categorical_cols=['grade', 'home_ownership', 'purpose']
"""
############# Organize & First Look ###################

cols=sorted(df.columns)
df[cols].head()
df.info()
############# Take sample and define dafualt ###################

df_sample = df.sample(10000)
df_sample = df_sample[cols]

df_sample['loan_status'].value_counts()

# Create default charactersitic: Default and 30+ dpd
print(df_sample['loan_status'].unique())
# Questions: What is charged off ?
#            Does not meet the credit policy ? 
pass_criteria = ['Current', 'Fully Paid', 'Charged Off', 'In Grace Period',
                    'Late (16-30 days)', 'Issued',
                    'Does not meet the credit policy. Status:Charged Off',
                    'Does not meet the credit policy. Status:Fully Paid']

df_sample['default'] = np.where(np.isin(df_sample['loan_status'],pass_criteria),0,1)



########### Function for Default Frequency
def Calc_Default_Freq_Cat(cat_var):
    return df_sample['default'].groupby(cat_var).sum() / cat_var.value_counts()


import matplotlib.pyplot as plt

###### Visualizations
####### Numerical Variables
####### Needs Auto Auto group function
"""
def hist_num_var(x,Title):
    
    #Idea 1 linspace pdf and get appropiate values
    #Idea 2 freq and amt get even groups
    #make a categorical variable the rebin categorical var
    
    kde  = st.gaussian_kde(x,bw_method=0.1)
    x_grid = np.linspace(x.min(), x.max(), 200)
    pdf = kde.evaluate(x_grid)
    fig, ax = plt.subplots()
    
    
    ax.hist(x,bins=44,normed=True)
    ax.set_title(Title)
    ax.plot(x_grid, pdf,color = 'red',label='pdf')
    ax.legend(loc='best')
    plt.show()
    

    return
"""
def hist_num_var(x,Title,numBin):
    
    #Idea 1 linspace pdf and get appropiate values
    #Idea 2 freq and amt get evenBgroups
    #ins = 4
    var, bins = pd.cut(x,bins = numBin ,retbins = True)
    var = var.sort_index().astype(str)
    auc = single_VAR_AUC(var)
    odf = Calc_Default_Freq_Cat(var).sort_index()
    freq_x = var.value_counts()

    digits = len(str(max(freq_x)))-1
    fig, ax1 = plt.subplots()
    ax1.bar(freq_x.index,freq_x, color='blue'
             ,alpha=1,label='Obs')
    ax1.set_title(Title + '\n AUC: ' + str(auc) +' %' )
    ax1.set_ylabel('Observations')
    ax1.yaxis.grid()
    plt.setp(ax1.get_xticklabels(), rotation=30, horizontalalignment='right')
    ax2 = ax1.twinx()
    ax2.plot(odf.index,odf,linestyle='--', marker='o', color='r',label='ODF')
    ax2.set_ylim(0,max(odf)*1.1)
    ax2.set_ylabel('Default Frequency')
    ax1.set_yticks(np.linspace(0, round(max(freq_x)*1.1,-digits), 5))
    ax2.set_yticks(np.linspace(0, round(max(odf)*1.1,3), 5))
    fig.legend(loc='lower right')
    plt.show()
    
    #Ad binned varibale to data model
    df_sample[var.name+'_bin'] = var

    return 



######## Calculate Single VAR AUC ##############
    
def hist_cat(x,Title):
    auc = single_VAR_AUC(x)
    odf = Calc_Default_Freq_Cat(x).sort_index()
    freq_x = x.value_counts().sort_index()
    digits = len(str(max(freq_x)))-1
    fig, ax1 = plt.subplots()
    ax1.bar(freq_x.index,freq_x, color='blue'
             ,alpha=1,label='Obs')
    ax1.set_title(Title + '\n AUC: ' + str(auc) +' %' )
    ax1.set_ylabel('Observations')
    ax1.yaxis.grid()
    plt.setp(ax1.get_xticklabels(), rotation=30, horizontalalignment='right')
    ax2 = ax1.twinx()
    ax2.plot(odf.index,odf,linestyle='--', marker='o', color='r',label='ODF')
    ax2.set_ylim(0,max(odf)*1.1)
    ax2.set_ylabel('Default Frequency')
    ax1.set_yticks(np.linspace(0, round(max(freq_x)*1.1,-digits), 5))
    ax2.set_yticks(np.linspace(0, round(max(odf)*1.1,3), 5))
    fig.legend(loc='lower right')
    plt.show()
    
    return

def single_VAR_AUC(cat_var):
    #Label Encoder necessary since stats
    le = preprocessing.LabelEncoder()
    #Declare Variables
    y = df_sample['default']
    X = smt.add_constant(le.fit_transform(cat_var))
    # Regression Analysis
    logit_model=sm.Logit(y,X)
    result=logit_model.fit(disp=0)
    
    X = le.fit_transform(cat_var)
    
    return round(roc_auc_score(y, result.predict())*100,2)



hist_cat(df_sample['emp_length'],'Employment Length in Years')

#funded amt
placeholder = hist_num_var(df_sample['int_rate'].str.replace('%','').astype(float),'interest rate',5)
cols

#Loan Amount
loan_amt = hist_num_var(df_sample['loan_amnt'],'Loan Amount',3)
# Debt to Income Ratio
dti = hist_num_var(df_sample['dti'],'Debt to Income Ratio',3)
# Total Number of Loans
hist_num_var(df_sample['tot_cur_bal'].fillna(0),'Total Current Balance',10)
#Purpose
hist_cat(df_sample['purpose'],'Loan Purpose')

#Grade
hist_cat(df_sample['grade'],'Lending Club Loan Grade')
#Home Ownership
hist_cat(df_sample['home_ownership'],'Home Ownerhsip')
#Termn
hist_cat(df_sample['term'],'Term')

hist_cat(df_sample['title'],'Loan Title')

test = df_sample['int_rate'].unique()


def Logistic_Regression_Analysis(model_vars,continous_vars = []):
    
    le = preprocessing.LabelEncoder()
    y = df_sample['default'].reset_index(drop=True)
    X = pd.DataFrame([])
    for var in model_vars:
        X[var]= le.fit_transform(df_sample[var])
    
    for i in continous_vars:
        X[i]= df_sample[i].reset_index(drop=True)
        
    #Add Constant
    X = smt.add_constant(X)
    # Regression Analysis
    logit_model=sm.Logit(y,X)
    result=logit_model.fit(disp=0)    
    
    from sklearn.metrics import roc_curve
    logit_roc_auc = roc_auc_score(y, result.predict())
    fpr, tpr, thresholds = roc_curve(y, result.predict())
    plt.figure()
    plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()
    print('AUC Score: '+str(logit_roc_auc))
    return result
##Binned results
    
model_vars = ['term','home_ownership','grade','purpose','emp_length','loan_amnt_bin','dti_bin']
results = Logistic_Regression_Analysis(model_vars)

##Unbinned results
model_vars = ['term','home_ownership','grade','purpose','emp_length',]
continous_vars = ['loan_amnt','dti']
results = Logistic_Regression_Analysis(model_vars,continous_vars)


cols






results.summary()
#Calculate Auc for Continous Vars No Binning

#le = preprocessing.LabelEncoder()

var = df_sample['loan_amnt']
#Declare Variables
y = df_sample['default']
X = smt.add_constant(var)
# Regression Analysis
logit_model=sm.Logit(y,X)
result=logit_model.fit(disp=0)

round(roc_auc_score(y, result.predict())*100,2)






## Implement Naive Binning
#Idea 1 linspace pdf and get appropiate values
#Idea 2 freq and amt get even groups
#make a categorical variable the rebin categorical var
"""
from scipy import integrate
kde  = st.gaussian_kde(var,bw_method=0.1)
x_grid = np.linspace(var.min(), var.max(), 200)
pdf = kde.evaluate(x_grid)
cdf = integrate.cumtrapz(pdf, x_grid, initial=0)

np.where(cdf.any() in [0,0.1,0.2,0.3])
"""
bins = 4
from string import ascii_uppercase
letters = list(ascii_uppercase[0:bins])
var, bins = pd.cut(df_sample['loan_amnt'],bins = bins,
                   labels=range(0,bins),retbins = True)

var, bins = pd.cut(df_sample['loan_amnt'],bins = bins ,retbins = True)

le = preprocessing.LabelEncoder()
#Declare Variables
y = df_sample['default']
X = smt.add_constant(le.fit_transform(var))
# Regression Analysis
logit_model=sm.Logit(y,X)
result=logit_model.fit(disp=0)

round(roc_auc_score(y, result.predict())*100,2)



######### Map of US Data
######### ToDo Construct the same for for Loan_Amount an Number of loans 
from collections import OrderedDict

cross_def = pd.crosstab(df_sample['addr_state'],df_sample['default'])

cross_def.columns=['Good Loans','Bad Loans']
def_ratio = (cross_def['Bad Loans']/(cross_def['Bad Loans']+cross_def['Good Loans'])).values.tolist()
number_of_badloans = cross_def['Bad Loans'].values.tolist()
state_codes = sorted(df_sample['addr_state'].unique().tolist())
risk_data = OrderedDict([('state_codes', state_codes),
                         ('default_ratio', def_ratio),
                         ('badloans_amount', number_of_badloans)])
                     
risk_df = pd.DataFrame.from_dict(risk_data)
risk_df = risk_df.round(decimals=3)
risk_df.head()


# Now it comes the part where we plot out plotly United States map
#import plotly.plotly as py
#import plotly.graph_objs as go


for col in risk_df.columns:
    risk_df[col] = risk_df[col].astype(str)
    
scl = [[0.0, 'rgb(202, 202, 202)'],[0.2, 'rgb(253, 205, 200)'],[0.4, 'rgb(252, 169, 161)'],\
            [0.6, 'rgb(247, 121, 108  )'],[0.8, 'rgb(232, 70, 54)'],[1.0, 'rgb(212, 31, 13)']]

risk_df['text'] = risk_df['state_codes'] + '<br>' +\
'Number of Bad Loans: ' + risk_df['badloans_amount'] + '<br>' + \
'Observed Default Frequency: ' + risk_df['default_ratio'] + '%' 



data = [ dict(
        type='choropleth',
        colorscale = scl,
        autocolorscale = False,
        locations = risk_df['state_codes'],
        z = risk_df['default_ratio'], 
        locationmode = 'USA-states',
        text = risk_df['text'],
        marker = dict(
            line = dict (
                color = 'rgb(255,255,255)',
                width = 2
            ) ),
        colorbar = dict(
            title = "%")
        ) ]


layout = dict(
    title = 'Lending Club Customer Default Rates <br> (State by State)',
    geo = dict(
        scope = 'usa',
        projection=dict(type='albers usa'),
        showlakes = True,
        lakecolor = 'rgb(255, 255, 255)')
)

fig = dict(data=data, layout=layout)
py.plot(fig, filename='d3-cloropleth-map')



