[<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/banner.png" width="888" alt="Visit QuantNet">](http://quantlet.de/)

## [<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/qloqo.png" alt="Visit QuantNet">](http://quantlet.de/) **brierScore** [<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/QN2.png" width="60" alt="Visit QuantNet 2.0">](http://quantlet.de/)

```yaml

Name of QuantLet : brierScore

Published in : SPL

Description : Calculates Brier Score for the prediction model

Keywords : 'logistic regression, score, brier, accuracy'

Author : Tobias Blücher, Niklas Kudernak, Sydney Richards

```



### Pyhton Code:
```python


# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 12:02:22 2018

@author: sydma
"""

########## Statisitcal Programming Languagues SS 2018 ##########
########## Project: Credit Risk Scorecard in Python   ##########
########## Members: Tobias Blücher, Niklas Kudernak  ##########
##########          Sydney Richards                   ##########
########## 
# Project Design: To create a Scorecard to classify Customer   #
# Loan applications according to the charactheristics of the   #
# applicant. While the grouping shall be done using a Logistic #
# Regression. The scorecard that is develpod should be usuable #
# by non statisticians, since Lending Club is a peer-to-peer   #
# platform.

#### Dataset: Kaggle Competition kaggle datasets download -d wendykan/lending-club-loan-data
#### Use the Kaggle API to download dataset files.
#### https://github.com/Kaggle/kaggle-api 
#### Link: https://www.kaggle.com/wendykan/lending-club-loan-data/data


#### Packages Required
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn import preprocessing 
from sklearn.metrics import brier_score_loss
import statsmodels.api as sm
import statsmodels.tools as smt


######################## Step 1: Data Import ##################################                

### Set file path to local directory
data_path = 'C:\\Users\\sydma\\Dropbox\\Uni Sach\Master\\SoSe_18\\Statistical Programming Languages\\Topic\\Loan_Club\\'
### Connect to SQL engine
engine = create_engine('sqlite:///' + data_path +'database.sqlite')
### Load Dataset
df = pd.read_sql('SELECT * FROM loan' ,engine)

### Organize & First Look 
cols=sorted(df.columns)
df[cols[1:25]].head()
df.info()
### Drop iffy loan_status

status = ['Issued','Does not meet the credit policy. Status:Charged Off',
                    'Does not meet the credit policy. Status:Fully Paid']
df = df[~df['loan_status'].isin(status)]

############# Take sample and define dafualt ###################

######################## Step 2: Create Sample ################################  

### create sample of 50000 obs with random_state(seed)
df_sample =df.sample(50000,random_state = 1)
### Only select useful columns
df_sample = df_sample[cols]

### Convert Date Cols
df_sample['last_pymnt_d'] = pd.to_datetime(df_sample['last_pymnt_d'])
df_sample['earliest_cr_line'] = pd.to_datetime(df_sample['earliest_cr_line'])
### latest obervation in dataset is '31/01/2016/' Calc Credit History
df_sample['final_snp_date'] =  pd.to_datetime('31/01/2016/')
### Length of Credit History in Years
df_sample['cr_hist_yr']=(df_sample['final_snp_date'].dt.to_period('M') - df_sample['earliest_cr_line'].dt.to_period('M')) / 12
### Convert to float 
df_sample['cr_hist_yr'] = df_sample['cr_hist_yr'].astype(float)


### Create default charactersitic: Default and 30+ dpd
print(df_sample['loan_status'].unique())
### Questions: What is charged off ? 
### Charged Off means in goes into collections 150+ dpd default
### Does not meet the credit policy ? 
pass_criteria = ['Current', 'Fully Paid','In Grace Period',
                    'Late (16-30 days)', 'Issued','Late (31-120 days)']

df_sample['default'] = np.where(np.isin(df_sample['loan_status'],pass_criteria),0,1)




########################  Brier Score  ############################## 

### The Brier Score is a proper score function which measures the
### accuracy of probabilistic predictions in a set of mutually
### exclusive outcomes i.e. default, non-default

model_vars = ['term','home_ownership','grade','purpose','emp_length',]
continous_vars = ['funded_amnt','dti']    


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
    
y_true = df_sample['default']
y_pred = result.predict()


print(brier_score_loss(y_true, y_pred))
```