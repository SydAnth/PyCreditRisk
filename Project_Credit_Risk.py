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
import matplotlib.pyplot as plt



import plotly.plotly as py
import plotly.graph_objs as go
import plotly 


plotly.tools.set_credentials_file(username='SPL2018SS', api_key='X8qHACTMQmxmmsnQNfmD')

############# Import Data ###################

data_path = 'C:\\Users\\sydma\\Dropbox\\Uni Sach\Master\\SoSe_18\\Statistical Programming Languages\\Topic\\Loan_Club\\'
engine = create_engine('sqlite:///' + data_path +'database.sqlite')
df = pd.read_sql('SELECT * FROM loan' ,engine)
"""
numerical_cols=['sub_grade_num', 'short_emp', 'emp_length_num','dti', 'payment_inc_ratio', 'delinq_2yrs', \
                'delinq_2yrs_zero', 'inq_last_6mths', 'last_delinq_none', 'last_major_derog_none', 'open_acc',\
                'pub_rec', 'pub_rec_zero','revol_util']

categorical_cols=['grade', 'home_ownership', 'purpose']
"""
############# Organize & First Look ###################
cols=sorted(df.columns)
df[cols[1:25]].head()
df.info()
##### Drop iffy loan_status

status = ['Issued','Does not meet the credit policy. Status:Charged Off',
                    'Does not meet the credit policy. Status:Fully Paid']


df = df[~df['loan_status'].isin(status)]

############# Take sample and define dafualt ###################


df_sample = df.sample(10000,random_state = 1)
df_sample = df_sample[cols]

###################### Convert Date Cols
df_sample['last_pymnt_d'] = pd.to_datetime(df_sample['last_pymnt_d'])
df_sample['earliest_cr_line'] = pd.to_datetime(df_sample['earliest_cr_line'])


df_sample['final_snp_date'] =  pd.to_datetime('31/01/2016/')
#### Length of Credit History in Years
df_sample['cr_hist_yr']=(df_sample['final_snp_date'].dt.to_period('M') - df_sample['earliest_cr_line'].dt.to_period('M')) / 12

######### Drop URL Columns ##############

############# Check if Columns are filled and contain usefull info
suspect = []
for numeric_var in df_sample.columns:
     if  (df_sample[numeric_var].dtype == 'float64' and
     len(df_sample[numeric_var].unique()) < 100):
                suspect.append(numeric_var)
                
df_sample.columns                
vars_to_drop = [ 'acc_now_delinq',
                 'application_type', # too few joints to make a difference
                 'all_util',
                 'annual_inc_joint',
                 'desc',
                 'collection_recovery_fee', # no interest in collection
                 'collections_12_mths_ex_med',
                 'open_acc_6m',
                 'open_il_12m',
                 'open_il_24m',
                 'open_il_6m',
                 'open_rv_12m',
                 'open_rv_24m',
                 'policy_code',
                 'max_bal_bc',
                 'last_pymnt_d',#since we have loan_satus we don't need
                 'next_pymnt_d',#also not useful
                 'dti_joint',
                 'inq_fi',
                 'il_util',
                 'inq_last_12m',
                 'inq_last_6mths',
                 'total_bal_il',
                 'total_cu_tl',
                 'tot_coll_amt',
                 'url', 
                 'verification_status',
                 'verification_status_joint',
                 'member_id'#drop cause df['member_id'].nunique = obs
                 ,'id']
df_sample = df_sample.drop(vars_to_drop,axis=1)

# Create default charactersitic: Default and 30+ dpd
print(df_sample['loan_status'].unique())
# Questions: What is charged off ? 
             #Charged Off means in goes into collections 150+ dpd default
#            Does not meet the credit policy ? 
pass_criteria = ['Current', 'Fully Paid','In Grace Period',
                    'Late (16-30 days)', 'Issued','Late (31-120 days)']

df_sample['default'] = np.where(np.isin(df_sample['loan_status'],pass_criteria),0,1)


############## Dealing with Null Values #########################

nanCol = []
for col in df_sample.columns:
    if df_sample[col].hasnans == True:
            nanCol.append(col)
            
look_at = df_sample[nanCol]



### Fill Null with 0 since dlq has not happend
mth_dlq =    ['mths_since_last_delinq',
              'mths_since_last_major_derog',
              'mths_since_last_record',
              'mths_since_rcnt_il', 
              'tot_cur_bal', # is 0 if customer has no loans
              'total_rev_hi_lim'] # is 0 if customer has no loans
for col in mth_dlq:
   df_sample[col] = df_sample[col].fillna(0)
   
##### Employment Title recat as Unemployed   
df_sample['emp_title'] = df_sample['emp_title'].fillna('Unemployed')
   
##### Convert Utilization Ration to Float

df_sample['revol_util'] = df_sample['revol_util'].str.replace('%','').astype(float)
# Old Credit Line => 0 Denominator
df_sample[np.isnan(df_sample['revol_util'])]
df_sample['revol_util'] = df_sample['revol_util'].fillna(0)



################ Clustering of Variables ########################
# CLuster Eployment Title
# Why ? to many values





##################### Factorize Categorical Variables #####################
cat_vars = []

for col in df_sample.columns:
    if df_sample[col].dtype != 'float64':
        cat_vars.append(col)






############### Calculate Simple Correlation Matrix #############

import seaborn as sns


corr = df_sample.corr(method='pearson')


plt.subplots(figsize=(20,15))
sns.heatmap(corr)
### Can be dropped due to high correlations
vars_to_drop =  ['funded_amnt_inv',
                'installment',
                'loan_amnt',
                'total_pymnt',
                'total_pymnt_inv',
                'out_prncp_inv',
                'total_rec_int']
df_sample = df_sample.drop(vars_to_drop,axis=1)


############### Next Step Chi-Sq









########### Function for Default Frequency
def Calc_Default_Freq_Cat(cat_var):
    return df_sample['default'].groupby(cat_var).sum() / cat_var.value_counts()



###### Visualizations
####### Numerical Variables
####### Needs Auto Auto group function
"""

# define a binning function
def mono_bin(Y, X, n = 20):
  # fill missings with median
  X2 = X.fillna(np.median(X))
  r = 0
  while np.abs(r) < 1:
    d1 = pd.DataFrame({"X": X2, "Y": Y, "Bucket": pd.qcut(X2, n)})
    d2 = d1.groupby('Bucket', as_index = True)
    r, p = stats.spearmanr(d2.mean().X, d2.mean().Y)
    n = n - 1
  d3 = pd.DataFrame(d2.min().X, columns = ['min_' + X.name])
  d3['max_' + X.name] = d2.max().X
  d3[Y.name] = d2.sum().Y
  d3['total'] = d2.count().Y
  d3[Y.name + '_rate'] = d2.mean().Y
  d4 = (d3.sort_index(by = 'min_' + X.name)).reset_index(drop = True)
  print "=" * 60
  print d4
 
mono_bin(data.bad, data.tot_income)
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

################ Step1: Univariate Analysis ##################################
    

####### Numerical Variables #####################################




#T'delinq_2yrs'
#The number of 30+ days past-due incidences of delinquency
#in the borrower's credit file for the past 2 years

df_sample['delinq_2yrs'].value_counts()
delinq_2yrs = hist_num_var(df_sample['delinq_2yrs'],'Delinquencies - 2years',8)
#Loan Amount
funded_amnt = hist_num_var(df_sample['funded_amnt'],'funded_amnt',8)
# Debt to Income Ratio
dti = hist_num_var(df_sample['dti'],'Debt to Income Ratio',3)
# Length of Credit History
cr_hist_yr = hist_num_var(df_sample['cr_hist_yr'],'Length Credit History',5)

#Purpose
hist_cat(df_sample['purpose'],'Loan Purpose')
#Length of Employment

hist_cat(df_sample['emp_length'],'Employment Length in Years')

#Grade
hist_cat(df_sample['grade'],'Lending Club Loan Grade')
#Home Ownership
hist_cat(df_sample['home_ownership'],'Home Ownerhsip')
#Termn
hist_cat(df_sample['term'],'Term')

#hist_cat(df_sample['title'],'Loan Title')

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
    
model_vars = ['term','home_ownership','grade','purpose','emp_length','funded_amnt_bin','dti_bin']
results = Logistic_Regression_Analysis(model_vars)



##Unbinned results
model_vars = ['term','home_ownership','grade','purpose','emp_length',]
continous_vars = ['funded_amnt','dti']
results = Logistic_Regression_Analysis(model_vars,continous_vars)




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



