[<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/banner.png" width="888" alt="Visit QuantNet">](http://quantlet.de/)

## [<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/qloqo.png" alt="Visit QuantNet">](http://quantlet.de/) **PyCreditRisk** [<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/QN2.png" width="60" alt="Visit QuantNet 2.0">](http://quantlet.de/)

```yaml

Name of QuantLet : PyCreditRisk

Published in : SPL

Description : Project Design: To create a Scorecard to classify Customer   
	      Loan applications according to the charactheristics of the   
	      applicant. While the grouping shall be done using a Logistic 
              Regression. The scorecard that is develpod should be usuable 
              by non statisticians, since Lending Club is a peer-to-peer   
              platform.
	      
              Additionally the Exercise Sheets from the R-Tutorial were 
              solved in Python as a Reference for future courses.


Keywords : 'logistic regression, credit risk,python, risk,exercises,tutorial'

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
import scipy.stats as st

import statsmodels.api as sm
import statsmodels.tools as smt
import matplotlib.pyplot as plt



import itertools
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import brier_score_loss
from sklearn import preprocessing 
from sklearn.metrics import roc_auc_score
from collections import OrderedDict



```