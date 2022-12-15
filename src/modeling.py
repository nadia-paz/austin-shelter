import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import src.wrangle as wr



from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, ConfusionMatrixDisplay

'''
intake -> when features used in the models are exclusevly from intake data set
whole -> features used are from both data sets
'''

############## GLOBAL VARIABLES #############
seed = 2912
models_to_pick = 2
number_of_features = 4

# get train/validate/test data sets & targets
X_train, X_validate, X_test, y_train, y_validate, y_test = wr.split_data(wr.get_shelter_data(), explore=False)
# scale data sets
train_scaled, validate_scaled, test_scaled = wr.scale_quantile(X_train, X_validate, X_test)
# set the baseline
baseline = round(y_train.value_counts(normalize=True).max(), 2)

####### FEATURE COMBINATIONS ###########
intake_only = ['animal_type_Cat', 'animal_type_Dog', 'animal_type_Guinea Pig',
       'animal_type_Livestock', 'animal_type_Rabbit', 'animal_type_Wild',
       'intake_type_Euthanasia Request', 'intake_type_Owner Surrender',
       'intake_type_Public Assist', 'intake_type_Stray',
       'intake_type_Wildlife', 'intake_condition_Agonal',
       'intake_condition_Behavior', 'intake_condition_Feral',
       'intake_condition_Injured', 'intake_condition_Med Attn',
       'intake_condition_Medical', 'intake_condition_Neonatal',
       'intake_condition_Normal', 'intake_condition_Nursing',
       'intake_condition_Other', 'intake_condition_Pregnant',
       'intake_condition_Sick', 'intake_condition_Unknown', 'female', 'male',
       'intact', 'is_steril', 'age_on_check_in',
       'mixed_breed', 'domestic_breed','pitbull']

# all columns
whole_set = X_train.columns.to_list()
# no breed dummies
intake_no_breed = intake_only[:-3] # take away breed dummies
whole_set_no_breed = whole_set[:-3] 
# no breed and no engineered features
intake_no_extra = intake_only[:-4] 
whole_set_no_extra = whole_set[:-7]
# no engineered features but with breed dummies
intake_with_breed = intake_no_extra + ['mixed_breed', 'domestic_breed','pitbull']
whole_set_with_breed = whole_set_no_extra + ['mixed_breed', 'domestic_breed','pitbull']

# dictionaries to keep features and their names
intake_names= {
              1:'intake_only', 
              2:'intake_no_breed', 
              3:'intake_no_extra',
              4:'intake_with_breed'
            }
whole_names = {
              1:'whole_set',
              2:'whole_set_no_extra',
              3:'whole_set_no_breed',
              4:'whole_set_with_breed'
            }
intake_dict= {
              1:intake_only, 
              2:intake_no_breed, 
              3:intake_no_extra,
              4:intake_with_breed
            }
whole_dict = {
              1:whole_set,
              2:whole_set_no_extra,
              3:whole_set_no_breed,
              4:whole_set_with_breed
            }

# data frames to keep the best results and predictions
best_intake = pd.DataFrame(columns=['model_name', 'features', 'parameters', 'train_score', 'validate_score'])
best_whole = pd.DataFrame(columns=['model_name', 'features', 'parameters', 'train_score', 'validate_score'])
predictions_train = pd.DataFrame(y_train)
predictions_validate = pd.DataFrame(y_validate)
predictions_train_intake = pd.DataFrame(y_train)
predictions_validate_intake = pd.DataFrame(y_validate)


######### FUNCTIONS ##########

def gen_decision_trees(intake=True):
    '''
    the function without parameters run Decision Tree Classifier on intake features, intake=False -> on the whole set
    returns a data frame with models_to_pick number of best performing models
    '''
    # dataframe to keep scores
    scores_dt = pd.DataFrame(columns=['model_name', 'features', 'parameters', 'train_score', 'validate_score'])
    # define which dictionary/names to use
    if intake:
        features_dict = intake_dict
        features_names = intake_names
    else:
        features_dict = whole_dict
        features_names = whole_names
    
    # run models
    for key in range(1,number_of_features):
        for i in range(3, 7):
            model = DecisionTreeClassifier(max_depth = i, random_state=seed)
            model.fit(X_train[features_dict[key]], y_train)
            
            #calculate scores
            score = round(model.score(X_train[features_dict[key]], y_train), 3)
            validate = round(model.score(X_validate[features_dict[key]], y_validate), 3)

            scores_dt.loc[len(scores_dt.index)] = ['Decision Tree', features_names[key], i, score, validate]
            
            col_name = 'DT_'  + str(i) + '_' + features_names[key]
            # save predictions
            if intake:
                predictions_train_intake[col_name] = model.predict(X_train[features_dict[key]])
                predictions_validate_intake[col_name] = model.predict(X_validate[features_dict[key]])
            else:
                predictions_train[col_name] = model.predict(X_train[features_dict[key]])
                predictions_validate[col_name] = model.predict(X_validate[features_dict[key]])

    return scores_dt.sort_values(by='train_score', ascending=False).head(models_to_pick)

def gen_random_forest(intake=True):
    '''
    the function without parameters run Random Forest Classifier on intake features, intake=False -> on the whole set
    returns a data frame with models_to_pick number of best performing models
    '''
    # dataframe to keep scores
    scores_rf = pd.DataFrame(columns=['model_name', 'features', 'parameters', 'train_score', 'validate_score'])
    # define which dictionary/names to use
    if intake:
        features_dict = intake_dict
        features_names = intake_names
    else:
        features_dict = whole_dict
        features_names = whole_names
    
    # run models
    for key in range(1,number_of_features):
        for i in range(4, 10):
            #build the model and fit X_train, y_train into it
            model = RandomForestClassifier(max_depth = i, random_state=seed)
            model.fit(X_train[features_dict[key]], y_train)

            #calculate scores
            score = round(model.score(X_train[features_dict[key]], y_train), 3)
            validate = round(model.score(X_validate[features_dict[key]], y_validate), 3)
            #save the information about the model and it's score to a dictionary
            scores_rf.loc[len(scores_rf.index)] = ['Random Forest', features_names[key], i, score, validate]
            
            col_name = 'RF_'  + str(i) + '_' + features_names[key]
            # save predictions
            if intake:
                predictions_train_intake[col_name] = model.predict(X_train[features_dict[key]])
                predictions_validate_intake[col_name] = model.predict(X_validate[features_dict[key]])
            else:
                predictions_train[col_name] = model.predict(X_train[features_dict[key]])
                predictions_validate[col_name] = model.predict(X_validate[features_dict[key]])
        
    return scores_rf.sort_values(by='train_score', ascending=False).head(models_to_pick)

def gen_logistic_regression(X_train=train_scaled, X_validate=validate_scaled, y_train=y_train, y_validate=y_validate, intake=True):
    '''
    the function without parameters run Logistic Regression Classifier on intake features, intake=False -> on the whole set
    returns a data frame with models_to_pick number of best performing models
    '''
    # dataframe to keep scores
    scores_lr = pd.DataFrame(columns=['model_name', 'features', 'parameters', 'train_score', 'validate_score'])
    # define which dictionary/names to use
    if intake:
        features_dict = intake_dict
        features_names = intake_names
    else:
        features_dict = whole_dict
        features_names = whole_names
    
    # run models
    for key in features_dict:
        #build the model and fit X_train, y_train into it
        model = LogisticRegression(random_state=seed, max_iter=700)
        model.fit(X_train[features_dict[key]], y_train)

        #calculate scores
        score = round(model.score(X_train[features_dict[key]], y_train), 3)
        validate = round(model.score(X_validate[features_dict[key]], y_validate), 3)
        
        #save the information about the model and it's score to a df
        scores_lr.loc[len(scores_lr.index)] = \
            ['Logistic Regression', features_names[key], 0, score, validate]

        # save predictions
        col_name = 'LR_' + features_names[key]
        if intake:
            predictions_train_intake[col_name] = model.predict(X_train[features_dict[key]])
            predictions_validate_intake[col_name] = model.predict(X_validate[features_dict[key]])
        else:
            predictions_train[col_name] = model.predict(X_train[features_dict[key]])
            predictions_validate[col_name] = model.predict(X_validate[features_dict[key]])
    return scores_lr.sort_values(by='train_score', ascending=False).head(models_to_pick)

def best_results_intake():
    '''
    the function runs all model on the intake features
    returns the best performing model. higher score on the validation set in combination 
    with the low difference between the train/validation scores
    '''
    best_intake = gen_decision_trees()
    best_intake = pd.concat([best_intake, gen_random_forest()], axis=0)
    best_intake = pd.concat([best_intake, gen_logistic_regression()], axis=0)
    best_intake['diff'] = round(best_intake.train_score - best_intake.validate_score, 4)
    return best_intake.sort_values(by=['validate_score', 'diff'], ascending=[False, True]).head(3)

def best_results_whole():
    '''
    the function runs all model on the features from intake and outcome sets
    returns the best performing model. higher score on the validation set in combination 
    with the low difference between the train/validation scores
    '''
    best_whole = gen_decision_trees(intake=False)
    best_whole = pd.concat([best_whole, gen_random_forest(intake=False)], axis=0)
    best_whole = pd.concat([best_whole, gen_logistic_regression(intake=False)], axis=0)
    best_whole['diff'] = round(best_whole.train_score - best_whole.validate_score,4)
    return best_whole.sort_values(by=['validate_score', 'diff'], ascending=[False, True]).head(3)

def run_best_model():
    '''
    the function runs Random Forest with max_dapth=9 on the test data set
    the function uses intake_only features
    '''
    #build the model and fit X_train, y_train into it
    model = RandomForestClassifier(max_depth = 9, random_state=seed)
    model.fit(X_train[intake_only], y_train)

    #calculate scores
    train_score = round(model.score(X_train[intake_only], y_train), 3)
    validate_score = round(model.score(X_validate[intake_only], y_validate), 3)
    test_score = round(model.score(X_test[intake_only], y_test), 3)
    yhat=model.predict(X_test[intake_only])
    return pd.DataFrame({'result':['Random Forest', train_score, validate_score, test_score]},\
                          index=['Model name', 'Train score', 'Validate score', 'Test score'])