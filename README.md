##### austin-shelter individual project
by Nadia Paz

## Austin Animal Shelter
#### Predicting the chances of animals of being adopted

## Project's Goal
Using the classification algorithms try to predict if the animal is going to be adopted.

## Project's Purpose
Create a model that potentially can help to evaluate an animal's chances to be adopted.

### Data source
Data was obtained from the austintexas.gov website on December 12, 2022. 
Links for the source:
- [Intakes](https://data.austintexas.gov/Health-and-Community-Services/Austin-Animal-Center-Intakes/wter-evkm)

- [Outcomes](https://data.austintexas.gov/Health-and-Community-Services/Austin-Animal-Center-Outcomes/9t4d-g238)

You can also find a cleaned data set in ```austin_animal_shelter.csv``` file.


## Steps to Reproduce
1) Clone this repo into your computer.
4) Run the ```austin_animal_shelter.ipynb``` file.

## Project's pipeline

### Aqcuire and prepare
1. Acquire the data.
2. Prepare the data for exploration and analysis. Clean the data in order that every row represents an observation and every columns represents a feature. Find out if there are some values missing and find a way to handle those missing values.
3. Change the data types if needed
4. Find if there are features which can be created to simplify the exploration process.
5. Determine which observations are outliers and handle them.
6. Create a data dictionary.
7. Split the data into 3 data sets: train, validate and test data (56%, 24%, and 20% respectively)

### Explore and pre-process
1. Explore the train data set through visualizations and statistical tests. 
2. Determine which variables have statistically significant relationships with our target variable ```outcome_type```. 
2. Make the exploration summary and document the main takeaways.
3. Impute the missing values if needed.
4. Pick the features which can help to build a good prediction model.
5. Identify if new features must be created to improve the model's accuracy.
6. Encode the categorical variables.
7. Split the target variable from the data sets.
8. Scale the data prior to modeling.

### Build a classification model
1. Pick the classification algorithms (*classifiers*) for creating the prediction model.
2. Create the models and evaluate classifiers using **accuracy** score on the train data set and pick best performing algorithms.
3. The good model is going to be the model that gets the score better than a baseline. 
4. Find out which model has the best performance: relatively high predicting power on the validation test and slight difference in the train and validation prediction results.
5. Make predictions for the test data set.
6. Evaluate the results.

### Report results from models built

### Draw conclusions

## Data Dictionary

