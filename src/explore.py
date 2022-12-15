import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import scipy.stats as stats

import src.wrangle as wr

# define the default font style and fontsize
plt.style.use('seaborn-whitegrid')
plt.rc('font', size=16)

# GLOBAL VARIABLES
target = 'outcome_type'
# get train data
train, _, _ = wr.split_data(wr.get_shelter_data())
############ dataframes for each outcome
adopted = train[train.outcome_type == 'Adoption']
transfered = train[train.outcome_type == 'Transfer']
euthanized = train[train.outcome_type == 'Euthanasia']
died = train[train.outcome_type == 'Died']
list_of_outcomes = [adopted, transfered, euthanized, died]
# dict to hold outcomes
dict_of_outcomes = {
    'Adopted':adopted,
    'Transfered':transfered,
    'Euthanized':euthanized,
    'Died':died   
}
############# dataframe for each animal type
cats = train[train.animal_type == 'Cat']
dogs = train[train.animal_type == 'Dog']
rabbits = train[train.animal_type == 'Rabbit']
others = train[(train.animal_type == 'Wild') | (train.animal_type == 'Bird') \
                                            | (train.animal_type == 'Guinea Pig') \
                                            | (train.animal_type == 'Livestock')]
animal_type_dict = {
    'Cats':cats,
    'Dogs':dogs,
    'Rabbits':rabbits,
    'Others':others
}

##############
# list of categorical variables
cat_vars = ['animal_type', 'sex', 'breed', 'color', 'intake_type', 'intake_condition', \
               'sterilized', 'sex_of_animal', 'mixed_breed', 'domestic_breed', 'pitbull', 'boxer']
# list of numerical columns
num_cols = ['times_in_shelter', 'age_in_days', 'age_in_months', 'age_in_years', 'days_at_shelter', 'age_on_check_in']

# alpha value 0.05 for the confidence level 95%
# to be used in stat tests
alpha = 0.05

#### variables for visualizations
piechart_labels = ['Adopted', 'Transfered', 'Euthanized', 'Died']
values = train.outcome_type.value_counts().tolist()
piechart_labels = ['Adopted', 'Transfered', 'Euthanized', 'Died']
flare = sns.color_palette('flare')
flare = sns.color_palette('Accent')
set2 = sns.color_palette('Set2')



######## FUNCTIONS ##########
##### helpers #####
def autopct_format(values):
    '''
    the function accept value_counts from outcome_type
    puts it in % format ready to use in pie charts
    '''
    def my_format(pct):
        total = sum(values)
        val = int(round(pct*total/100.0))
        return '{:.1f}%)'.format(pct, v=val)
    return my_format

############ VISUALIZATIONS ###################
def viz_animals_outcome():
    '''
    the function creates a pie chart that 
    '''
    plt.figure(figsize=(8, 8))
    plt.pie(values, labels=piechart_labels, explode=[0.01, 0.02, 0.03, 0.04], 
            colors=set2, autopct=autopct_format(values),
            shadow=True)
    plt.title('Animal\'s outcome')
    plt.show()

def viz_animal_type_vs_outcome():
    '''
    the functions shows pie charts for cats/ dogs/ rabbits and other animals
    showing their outcome types
    '''
    plt.figure(figsize=(16, 8))
    for i, key in enumerate(animal_type_dict):
        plt.subplot(2, 2, i+1)
        plt.title('\n' + key + '\n')
        v = animal_type_dict[key].outcome_type.value_counts().tolist()
        labels = animal_type_dict[key].outcome_type.value_counts().index.tolist()
        labels[3] = ''
        patches, text = plt.pie(v, labels=labels, explode=[0.01, 0.02, 0.03, 0.07], 
            colors=set2, shadow=True)

def sex_vs_outcome():
    '''
    returns 2 subplots that show outcomes for males/females and intact or sterilized animals
    '''
    plt.figure(figsize=(20, 6))
    plt.subplot(121)
    sns.histplot(data=train, x='sterilized', hue='outcome_type')
    plt.title('Neutered/Spayed vs Intact')
    plt.ylim(0, 20_000)
    plt.subplot(122)
    sns.histplot(data=train, x='sex_of_animal', hue='outcome_type')
    plt.title('Male vs Female')
    plt.ylim(0, 20_000)
    plt.show()

def age_vs_outcome():
    '''
    the function visualize the mean age of animal in month and outcomes
    '''
    plt.figure(figsize=(12, 6))
    graph = sns.barplot(y='age_in_months',  x='outcome_type', estimator=np.median, data=train,\
                        order = ['Adoption', 'Transfer', 'Euthanasia', 'Died'], palette='Accent')
    graph.axhline(train.age_in_months.median(), color = (0.4, 0.4, 0.4), label = 'median age')
    graph.axhline(train.age_in_months.mean(), color = 'red', label = 'mean age')
    plt.title('Age of the animal in months vs Outcome type\n')
    plt.text(-0.4, 20.1, 'max age = ' + str(train.age_in_years.max()) + 'years')
    plt.legend()
    plt.show()

def viz_age_type_outcome():
    '''
    the function shows the median age of each animal and highlights the outcome type for each animal type group
    '''
    plt.figure(figsize=(12, 6))
    sns.barplot(x='animal_type', y='age_in_months', hue='outcome_type', data=train, palette=set2)
    plt.title('Animal\'s age, type and the outcome in shelter \n')
    plt.show()

############ STATISTICS ###################

def stats_animal_type():
    '''
    the function runs a chi2 stat test checking if animal_type has a relation to outcome_type
    '''
    observed = pd.crosstab(train.animal_type, train[target])
    #run a chi squared test fot categorical data
    test = stats.chi2_contingency(observed)
    p_value = test[1]
    if p_value < alpha:
        print('Reject the Null Hypothesis')
        print('There is an association between animal_type and outcome_type')
    else:
        print('Fail to reject the Null Hypothesis')
        print('There is no association between animal_type and outcome_type')

def stats_sex_vs_outcome():
    '''
    the function runs a chi2 stat test checking if sex_of_animal has a relation to outcome_type
    '''
    observed = pd.crosstab(train.sex_of_animal, train[target])
    #run a chi squared test fot categorical data
    test = stats.chi2_contingency(observed)
    p_value = test[1]
    if p_value < alpha:
        print('Reject the Null Hypothesis')
        print('There is an association between sex_of_animal and outcome_type')
    else:
        print('Fail to reject the Null Hypothesis')
        print('There is no association between sex_of_animal and outcome_type')

def ttest_1s(col):
    '''
    the function runs a staistical 1 sample t-test to compare sample mean vs population mean
    '''
    for key in dict_of_outcomes:
        print(key)
        print('==========')
        t, p = stats.ttest_1samp(dict_of_outcomes[key][col], train[col].mean())
        if p < alpha:
            print('Reject the Null Hypothesis')
            print(f'There is a difference in means between the {key} mean and the population mean')
        else:
            print('Fail to reject the Null Hypothesis')
            print(f'There is no difference in means between the {key}  mean and the population mean and population mean')
        print()

def stats_compare_age(col):
    '''
    takes a numeric columns and compares if means differ amoung all outcome groups
    '''
    # Levene test
    p = stats.levene(adopted[col], transfered[col],\
                     euthanized[col], died[col])[1]
    if (p < alpha):
        print('Variances are different. Use an non-parametric Kruskal-Wallis test.')
    else:
        print('Variances are equal. Use a parametric ANOVA test')

    print()

    # Kruskal-Wallis test
    p_kr = stats.kruskal(adopted[col], transfered[col],\
                         euthanized[col], died[col])[1]
    if (p_kr < alpha):
        print('Reject the Null Hypothesis')
        print('There is a significant difference in age_on_check_in  between different outcome types')
    else:
        print('Fail to reject the Null Hypothesis')
        print('There is a significant difference in age_on_check_in  between different outcome types')

def test_num_values():
    '''
    runs a 1 sample t-test for numerical variables of each outcome
    '''
    num_p_values = pd.DataFrame(columns=['variable', 'outcome', 'p_value', 'significance'])
    for col in num_cols:
        for key in dict_of_outcomes:
            t, p = stats.ttest_1samp(dict_of_outcomes[key][col], train[col].mean())
            num_p_values.loc[len(num_p_values)] = [col, key, p, p<alpha]
    return num_p_values

def get_p_values(df, cat_vars, target):
    '''
    accepts a data frame and the list of categorical column names
    returns a data frame with p_values of all categorical variables
    '''

    #dictionary to hold names of the column and a p_value assotiated with it
    p_v = {}
    #for every column in category variables run a chi2 test
    for col in cat_vars:
        #create a crosstable
        observed = pd.crosstab(df[col], df[target])
        #run a chi squared test fot categorical data
        test = stats.chi2_contingency(observed)
        p_value = test[1]
        #add the result to the dictionary
        p_v[col] = p_value
        
        #transform a dictionary to Series and then to Data Frame
        p_values = pd.Series(p_v).reset_index()
        p_values.rename(columns = {'index':'Feature', 0:'P_value'}, inplace = True)
        p_values = p_values.sort_values(by='P_value')

        #add the column that shows if the result is significant
        p_values['is_significant'] = p_values['P_value'] < alpha
    
    return p_values