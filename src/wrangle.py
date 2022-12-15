import os
# DS libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import QuantileTransformer

'''
DATASETS downloaded on December 12, 2022 

Data source:
https://data.austintexas.gov/Health-and-Community-Services/Austin-Animal-Center-Intakes/wter-evkm

https://data.austintexas.gov/Health-and-Community-Services/Austin-Animal-Center-Outcomes/9t4d-g238

'''
seed = 2912
target = 'outcome_type'
# re-arrange columns
new_order = ['animal_id', 'animal_type', 'sex', 'breed', 'color', 'name',
    'intake_type', 'intake_condition', 'first_check_in',
'last_check_out', 'times_in_shelter',  'sterilized', 'sex_of_animal', 'date_of_birth', 'age_on_check_in', 'age_in_days',
'age_in_months', 'age_in_years', 'days_at_shelter', 'mixed_breed', 'domestic_breed', 'pitbull',
 'outcome_type', 'outcome_subtype']

def acquire():
    '''
    the function reads the data from 2 csv files
    with the intake and outcome information from Austin Animal shelter
    the function merges those 2 files into 1 dataframe and returns it
    '''

    
    # if *.csv file is not available locally, acquire data from SQL database
    # and write it as *.csv for future use
    
    # read intake information 
    intake = pd.read_csv('Austin_Animal_Center_Intakes.csv')
    
    # read outcome information
    outcome = pd.read_csv('Austin_Animal_Center_Outcomes.csv')
    
    # merge tables
    df = intake.merge(outcome, how='inner', on='Animal ID', suffixes=('_in', '_out'))
    
    return df

def clean_data(df):
    '''
    the function accepts a merged data from animal shelter
    and cleans it: drops duplicates, drops the columns that are identical in 2 tables
    returns a df with renames columns and changed data types
    '''
    # drop duplicates
    df.drop_duplicates(inplace=True) # 82 rows
    
    # rename columns 
    columns_to_rename = [] # creates a list to hold new column names
    for col in df.columns.tolist():
        # every column to lower case and replace space with underscore
        columns_to_rename.append(col.lower().replace(' ', '_'))
    # assign new names to the columns
    df.columns = columns_to_rename
    
    # drop identical columns and not needed column found_location
    columns_to_drop = ['monthyear_in', 'found_location', 'animal_type_in', 'sex_upon_intake', 'age_upon_intake', 
                   'breed_in', 'color_in', 'name_out', 'monthyear_out']
    df.drop(columns=columns_to_drop, inplace=True)
    
    # change the type to category
    to_categories = ['intake_type', 'intake_condition', 'sex_upon_outcome',
                'outcome_type', 'outcome_subtype', 'animal_type_out', 'breed_out', 'color_out']
    for col in to_categories:
        df[col] = pd.Categorical(df[col])
    
    # change the type to datetime
    date_columns = ['datetime_in', 'datetime_out', 'date_of_birth']
    for col in date_columns:
        df[col] = pd.to_datetime(df[col])
    
    # rename columns
    df.rename(columns={'name_in':'name',
                    'animal_type_out':'animal_type',
                    'sex_upon_outcome':'sex',
                    'age_upon_outcome':'age',
                    'breed_out':'breed',
                    'color_out':'color'
                  }, inplace=True)
    
    # remove negative age
    df = df[df.date_of_birth < df.datetime_in] # 250 rows
    # birthday = same day as check in. do not remove!!! where born in the shelter or right before found on the street
    # df = df[(df.datetime_in - df.date_of_birth).dt.days > 1]
    
    # re-order columns
    new_col_order = ['animal_id', 'animal_type', 'sex', 'age', 'breed', 'color', 'name', 'date_of_birth', 'datetime_in', 'datetime_out', 
                 'intake_type', 'intake_condition', 'outcome_type', 'outcome_subtype']
    return df[new_col_order]

def handle_duplicates(df):
    '''
    this function accepts a dataframe of animal shelter as a parameter
    ----
    some animals check-in and check-out in the shelter many times up to 1088 times
    this might have few reasons:
    a) an animal didn't fit a family and was returned
    b) animal escaped the owner and was returned to the owner
    c) animal was 'rented' for a weekend or for a walk
    
    those rows create 'noise'. I remove all rows where the animal stayed less than 2 days.
    
    for other duplicated id's I count how many times the animal `checked in` in the shelter
    and create a new column 'times_in_shelter'
    '''
    # leave only animals that stay > 1 day (other entries are showing walks with animal)
    df = df[(df.datetime_out - df.datetime_in).dt.days > 1] # 59295 rows
    
    # create a dataframe that holds duplicated values
    df_with_duplicates = df[df['animal_id'].duplicated()].copy()
    # create a dataframe with unique values only
    df_no_duplicates = df[~df['animal_id'].duplicated()].copy()
    
    # create a dictionry that holds how many times the animal stayed in the shelter
    # +1 because the same id is available in df_no_duplicates
    times_dict = (df_with_duplicates.animal_id.value_counts() + 1).to_dict()
    # creat a columns to hold number of times the animal went to the shelter
    df_no_duplicates['times_in_shelter'] = 0
    # map the values from the times_dict
    df_no_duplicates.times_in_shelter = df.animal_id.map(times_dict)
    # replace null values with 1 
    # (null means -> the animal doesn't have duplicate rows -> went to the shelter only once)
    # change the data type to int
    df_no_duplicates.times_in_shelter = df_no_duplicates.times_in_shelter.fillna(1).astype(int)

    ############### PICK THE MINIMUM CHECK IN DATE ######################
    # 6197 rows changed
    # min datetime_in values from the df_no_duplicates
    min_no_dupl = df_no_duplicates.groupby('animal_id').datetime_in.min()

    # min datetime_in values from the df_with_duplicates
    min_with_dupl = df_with_duplicates.groupby('animal_id').datetime_in.min()

    # concatenate series into data frame
    min_timeframe = pd.concat([min_no_dupl, min_with_dupl], axis = 1).copy()

    # drop null values (the ids that are not in both series)
    min_timeframe.dropna(inplace=True)

    # rename columns
    min_timeframe.columns = ['no_dupl', 'with_dupl']

    # compare dates
    min_timeframe['no_smaller'] = min_timeframe.no_dupl <= min_timeframe.with_dupl

    # create a new column that takes the smallest value
    min_timeframe['final_date_in'] = \
        np.where(min_timeframe.no_smaller, min_timeframe.no_dupl, min_timeframe.with_dupl)

    # drop unneeded columns
    min_timeframe.drop(columns=['no_dupl', 'with_dupl', 'no_smaller'], inplace=True)

    # convert into Series and then into a dictionary
    min_timeframe = min_timeframe.squeeze().to_dict()

    # create a column first_check_in that holds the minimum check-in date
    df_no_duplicates['first_check_in'] = np.nan
    df_no_duplicates.first_check_in = df.animal_id.map(min_timeframe)
    df_no_duplicates.first_check_in = df_no_duplicates.first_check_in.fillna(df_no_duplicates.datetime_in)

    ############### PICK THE MAXIMUM CHECK OUT DATE ######################
    # 6451 rows changed
    # max datetime_in values from the df_no_duplicates
    max_no_dupl = df_no_duplicates.groupby('animal_id').datetime_out.max()

    # max datetime_in values from the df_with_duplicates
    max_with_dupl = df_with_duplicates.groupby('animal_id').datetime_out.max()

    # concatenate series into data frame
    max_timeframe = pd.concat([max_no_dupl, max_with_dupl], axis = 1).copy()

    # drop null values (the ids that are not in both series)
    max_timeframe.dropna(inplace=True)

    # rename columns
    max_timeframe.columns = ['no_dupl', 'with_dupl']

    # compare dates
    max_timeframe['no_smaller'] = max_timeframe.no_dupl <= max_timeframe.with_dupl

    # create a new column that takes the biggest value
    max_timeframe['final_date_out'] = \
        np.where(max_timeframe.no_smaller, max_timeframe.with_dupl, max_timeframe.no_dupl)

    # drop unneeded columns
    max_timeframe.drop(columns=['no_dupl', 'with_dupl', 'no_smaller'], inplace=True)

    # convert into Series and then into a dictionary
    max_timeframe = max_timeframe.squeeze().to_dict()

    # create a column last_check_out that holds the maximum check-out date
    df_no_duplicates['last_check_out'] = np.nan
    df_no_duplicates.last_check_out = df.animal_id.map(max_timeframe)
    df_no_duplicates.last_check_out = df_no_duplicates.last_check_out.fillna(df_no_duplicates.datetime_out)

    # drop 'datetime_in', 'datetime_out' columns
    df_no_duplicates.drop(columns=['datetime_in', 'datetime_out'], inplace=True)
    
    return df_no_duplicates

def add_features(df):
    '''
    the function takes a dataframe of animal shelter as a parameter
    adds more features for the exploration
    
    '''
    # spayed, neutered, intact, unknown
    df['sterilized'] = df.sex.str.split(' ', expand=True).iloc[:, 0]
    # male, female, unknown
    df['sex_of_animal'] = df.sex.str.split(' ', expand=True).iloc[:, 1]
    # fill nulls with unknown
    df.sex_of_animal = df.sex_of_animal.fillna('Unknown')
    # fill nulls with unknown
    df.sterilized = df.sterilized.fillna('Unknown')
    # calculate age in days
    df['age_in_days'] = (df.last_check_out - df.date_of_birth).dt.days
    # calculate age in months
    df['age_in_months'] = (df.age_in_days / 30).astype(int)
    # calculate age in years
    df['age_in_years'] = (df.age_in_days / 365).astype(int)
    # age on check in
    df['age_on_check_in']=(df.first_check_in - df.date_of_birth).dt.days
    # if the animal is not cat/dog -> remove ' Mix' in the end
    df.breed = np.where(((df.animal_type == 'Other') & df.breed.str.contains(' Mix')), \
                    df.breed.str.strip(' Mix'), df.breed )
    # dummy if the breed is mixed
    df['mixed_breed'] = np.where((df.breed.str.contains(' Mix') | (df.breed.str.contains('/'))), 1, 0 )
    # dummy for domestic breeds
    df['domestic_breed'] = np.where(df.breed.str.contains('Domestic'), 1, 0)
    # dummy for Unknown breed
    #df['unknown_breed'] = np.where(df.breed.str.contains('Unknown'), 1, 0)
    # dummy for Pit Bull and Boxer
    df['pitbull'] = np.where(df.breed == 'Pit Bull', 1, 0).astype('uint8')
    #df['boxer'] = np.where(df.breed == 'Boxer', 1, 0).astype('uint8')
    # how many days in shelter the animal have spent before the final outcome
    df['days_at_shelter'] = (df.last_check_out - df.first_check_in).dt.days
    # replace animal_type with the breed name if the type is 'Other'
    df.animal_type = np.where(df.animal_type == 'Other', df.breed, df.animal_type)
    # create a dictionary that holds all bunnies breeds
    bunnie_cond = df.animal_type.str.contains('Rabbit') | \
            df.animal_type.str.contains('Dutch') | \
            df.animal_type.str.contains('Lop') | \
            df.animal_type.str.contains('Rex') | \
            df.animal_type.str.contains('Hare') | \
            df.animal_type.str.contains('Flemish') | \
            df.animal_type.str.contains('Dwarf') | \
            df.animal_type.str.contains('Angora-') | \
            df.animal_type.str.contains('Hotot')
    df.animal_type = np.where(bunnie_cond, 'Rabbit', df.animal_type)
    bunnies = {'Jersey Wooly':'Rabbit', 'Silver':'Rabbit', 'Californian':'Rabbit',\
          'New Zealand Wht':'Rabbit', 'Checkered Giant':'Rabbit', 'American Sable':'Rabbit', 'Polish':'Rabbit',\
          'Beveren':'Rabbit', 'Lionhead':'Rabbit', 'American':'Rabbit', 'Himalayan':'Rabbit',\
           'Cottontail':'Rabbit', 'Rhinelander':'Rabbit', 'Harlequin':'Rabbit', 'English Spot':'Rabbit', 'Havana':'Rabbit',\
           'Britannia Petit':'Rabbit'
          }
    # replace all bunnies breeds with Rabbit
    df.animal_type = df.animal_type.replace(bunnies)
    # put unknown to all animals that are not cats or dogs
    df.breed = np.where(((df.animal_type == 'Cat') | (df.animal_type == 'Dog')), df.breed, 'Unknown')
    # keep only some animals, the rest is Wild
    animals_cond = (df.animal_type == 'Dog') | \
                (df.animal_type == 'Cat') | \
                (df.animal_type == 'Rabbit') | \
                (df.animal_type == 'Bird') | \
                (df.animal_type == 'Guinea Pig') | \
                (df.animal_type == 'Livestock')
    df.animal_type = np.where(animals_cond, df.animal_type, 'Wild')

    
    # remove age column
    df.drop(columns='age', inplace=True)
    
    return df

def drop_values(df):
    # drop null from the target -> 10 rows
    df = df[~df.outcome_type.isnull()].copy()
    # drop rare outcomes < 300 rows
    df = df[~((df.outcome_type == 'Lost') | (df.outcome_type == 'Stolen') | (df.outcome_type =='Disposal') \
        | (df.outcome_type == 'Missing'))].copy()
    # drop those who are in the shelter > 20 times
    df = df[df.times_in_shelter <= 20].copy() # 7- rows

    cond_transfer = (df.outcome_type == 'Transfer') | (df.outcome_type == 'Relocate')
    cond_adopt = (df.outcome_type == 'Adotion') | (df.outcome_type == 'Return to Owner') | \
            ((df.outcome_type == 'Rto-Adopt'))
    df.outcome_type = np.where(cond_transfer, 'Transfer', df.outcome_type)
    df.outcome_type = np.where(cond_adopt, 'Adoption', df.outcome_type)

    return df

########### FUNCTION TO CHANGE DATA TYPES WHEN DATA PULLED FROM THE *CSV 
def change_dtypes(df):
    '''
    the function accepts a cleaned dataframe of austin animal shelter
    changes data types to category, datetime and uint8
    rearranges columns and returns updated dataframe
    '''
    # change objects to category
    to_categories = ['animal_type', 'sex', 'intake_type', 'intake_condition', 'outcome_type', 'outcome_subtype', \
                  'sterilized', 'sex_of_animal']
    for col in to_categories:
            df[col] = pd.Categorical(df[col])
    # change dates to datetime
    to_date = ['date_of_birth', 'first_check_in', 'last_check_out']
    for col in to_date:
        df[col] = pd.to_datetime(df[col])
    # change small numbers to uint8
    to_uint8 = ['times_in_shelter', 'age_in_years', 'mixed_breed', 'domestic_breed', 'pitbull']
    for col in to_uint8:
        df[col] = df[col].astype('uint8')
    
    return df

######### RUN ALL FUNCTIONS ##################
def get_shelter_data():
    '''
    the function runs all functions above and returns a data set ready for split and exploration
    '''
    filename = 'data/austin_animal_shelter.csv'
    # if the clean file is available
    if os.path.isfile(filename):
        df = pd.read_csv(filename)
        df = change_dtypes(df)
    else:
        df = acquire()
        df = clean_data(df)
        df = handle_duplicates(df)
        df = add_features(df)
        df = drop_values(df)
        # save data to *csv file
        df.to_csv(filename, index_label = False)

    return df[new_order]

########## CREATE DUMMIES #######################
def dummies(df):
    '''
    the function accepts a dataframe as a parameter
    drops the columns that don't go into modeling
    creates dummies for categorical variables
    returns a dataframe ready for the split before the modeling
    '''
    columns_to_drop = ['animal_id', 'breed', 'sex', 'color', 'name', \
        'first_check_in', 'last_check_out','date_of_birth', 'age_in_months', 'age_in_years', 'outcome_subtype']
    df.drop(columns = columns_to_drop, inplace=True)
    
    # create dummies for sterilized and sex_of_animal. Both have 'Unknown' in same rows
    df.insert(0,'is_steril','')
    df['is_steril'] = np.where((df.sterilized == 'Neutered') | (df.sterilized == 'Spayed'), 1, 0).astype('uint8')
    df.insert(0,'intact','')
    df['intact'] = np.where(df.sterilized == 'Intact', 1, 0).astype('uint8')
    df.insert(0,'male','')
    df['male'] = np.where(df.sex_of_animal == 'Male', 1, 0).astype('uint8')
    df.insert(0,'female','')
    df['female'] = np.where(df.sex_of_animal == 'Female', 1, 0).astype('uint8')
    # unknows sex / sterilized is the same, it will get 00
    df.drop(columns=['sex_of_animal', 'sterilized'], inplace=True)
    
    # create dummies for 
    to_dummies = ['animal_type', 'intake_type','intake_condition']
    df = pd.concat(\
        [pd.get_dummies(df[to_dummies], drop_first=True), df], axis=1)
    df.drop(columns=to_dummies, inplace=True)
    
    return df

########### SPLIT BEFORE THE EXPLORATION ########

def split_3(df):
    '''
    This function takes in a dataframe and splits it into 3 data sets
    Test is 20% of the original dataset, validate is .30*.80= 24% of the 
    original dataset, and train is .70*.80= 56% of the original dataset. 
    The function returns, in this order, train, validate and test dataframes. 
    '''
    #split_db class verision with random seed
    train_validate, test = train_test_split(df, test_size=0.2, 
                                            random_state=seed, stratify=df[target])
    train, validate = train_test_split(train_validate, test_size=0.3, 
                                       random_state=seed, stratify=train_validate[target])
    return train, validate, test

def split_data(df, explore=True):
    '''
    the function accepts a dataframe as a parameter
    splits according to the purpose
    for the exploration returns train, validate, test
    for modeling it drops unneeded columns, creates dummis, and returns
    6 values X_train, y_train ...
    '''

    if explore:
        return split_3(df)
    else:
        df = dummies(df)
        train, validate, test = split_3(df)
        return train.iloc[:, :-1], validate.iloc[:, :-1], test.iloc[:, :-1], \
            train[target], validate[target], test[target]

def scale_quantile(train, validate, test):
    '''
    accepts train, validate, test data sets
    scales the data in each of them
    returns transformed data sets
    '''
    col = ['times_in_shelter', 'age_on_check_in', 'age_in_days', 'days_at_shelter']
    # create a quantile transformer  
    qt = QuantileTransformer(output_distribution='normal')
    qt.fit(train[col])
    train[col] = qt.transform(train[col])
    validate[col] = qt.transform(validate[col])
    test[col] = qt.transform(test[col])
    
    return train, validate, test
