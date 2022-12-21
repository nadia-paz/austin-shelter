import pandas as pd
import numpy as np


from sklearn.preprocessing import  PolynomialFeatures
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_squared_error, explained_variance_score

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import xgboost as xgb

# linear regressions
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor, Ridge

# non-linear regressions
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

import src.wrangle as wr

############## GLOBAL VARIABLES ###########
seed = 2912 # random seed for random_states

# load shelter data
df = wr.get_shelter_data()
# create dummies for categorical variables
df = wr.dummies(df)
# drop unneeded columns
df.drop(columns=['age_in_days', 'times_in_shelter', 'mixed_breed', 'domestic_breed', 'pitbull', 'outcome_type'], 
        inplace=True)

# split the data
train_validate, test = train_test_split(df, test_size=0.2, 
                                        random_state=seed)
train, validate = train_test_split(train_validate, test_size=0.3, 
                                   random_state=seed)

X_train = train.iloc[:, :-1].copy()
X_validate = validate.iloc[:, :-1].copy()
X_test = test.iloc[:, :-1].copy()
# scale age_on_check_in columns with the standard scaler
scaler = StandardScaler()
X_train[['age_on_check_in']] = scaler.fit_transform(X_train[['age_on_check_in']])
X_validate[['age_on_check_in']] = scaler.transform(X_validate[['age_on_check_in']])
X_test[['age_on_check_in']] = scaler.transform(X_test[['age_on_check_in']])
# target variable for train, validate and test sets
y_train = train.iloc[:, -1:].copy()
y_validate = validate.iloc[:, -1:].copy()
y_test = test.iloc[:, -1:].copy()

# baseline 65.729545164 days
baseline = y_train.mean()
