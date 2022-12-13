import pandas as pd

'''
DATASETS downloaded on December 12, 2022 

Data source:
https://data.austintexas.gov/Health-and-Community-Services/Austin-Animal-Center-Intakes/wter-evkm

https://data.austintexas.gov/Health-and-Community-Services/Austin-Animal-Center-Outcomes/9t4d-g238

'''

def acquire():
    '''
    the function reads the data from 2 csv files
    with the intake and outcome information from Austin Animal shelter
    the function merges those 2 files into 1 dataframe and returns it
    '''
    # read intake information 
    intake = pd.read_csv('Austin_Animal_Center_Intakes.csv')
    
    # read outcome information
    outcome = pd.read_csv('Austin_Animal_Center_Outcomes.csv')
    
    # merge tables, to 
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
    
    # re-order columns
    new_col_order = ['animal_id', 'animal_type', 'sex', 'age', 'breed', 'color', 'name', 'date_of_birth', 'datetime_in', 'datetime_out', 
                 'intake_type', 'intake_condition', 'outcome_type', 'outcome_subtype']
    return df[new_col_order]

