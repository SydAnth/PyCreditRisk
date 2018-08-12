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
import statsmodels.api as sm
import statsmodels.tools as smt
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix

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



######################## Step 5: Quality Checks  ############################## 
# Finally, we use additional performance measures to evaluate 		#
# the predictive power of our model. We use the Brier Score   		#
# which is used for predictions of discrete mutually          		#
# excludable events. We also plot a confusion_matrix          		#
# In the field of machine learning and specifically the 			#
# problem of statistical classification,a confusion matrix, 		#
# also known as an error matrix is a specific table layout 		#
#that allows visualization of the performance of an algorithm,		#
# typically a supervised learning one 								   #
# (in unsupervised learning it is usually called a matching matrix).# 
# Each row of the matrix represents the instances in a				#
# each column represents the instances in an actual class 			#
# (or vice versa) The name stems from the fact that it makes 		#
# it easy to see if the system is confusing two classes 			#
# (i.e. commonly mislabeling one as another)  						#


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
y_pred = [ 0 if x < 0.05 else 1 for x in result.predict()]


def plot_confusion_matrix(cm, classes,
                         normalize=False,
                         title='Confusion matrix',
                         cmap=plt.cm.Blues):
   """
   This function prints and plots the confusion matrix.
   Normalization can be applied by setting normalize=True.
   """
   if normalize:
       cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
       print('Normalized confusion matrix')
   else:
       print('Confusion matrix, without normalization')

   print(cm)

   plt.imshow(cm, interpolation='nearest', cmap=cmap)
   plt.title(title)
   plt.colorbar()
   tick_marks = np.arange(len(classes))
   plt.xticks(tick_marks, classes, rotation=45)
   plt.yticks(tick_marks, classes)

   fmt = '.2f' if normalize else 'd'
   thresh = cm.max() / 2.
   for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
       plt.text(j, i, format(cm[i, j], fmt),
                horizontalalignment='center',
                color='white' if cm[i, j] > thresh else 'black')

   plt.tight_layout()
   plt.ylabel('True label')
   plt.xlabel('Predicted label')

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_true, y_pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['non-default','default'],
                     title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['non-default','default'], normalize=True,
                     title='Normalized confusion matrix')

plt.show()