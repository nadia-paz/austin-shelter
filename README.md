##### austin-shelter individual project
by Nadia Paz

## Austin Animal Center
#### Can we predict if the animal going to be adopted?

## Project's Goal
Find the features that will help us to predict our target variable - outcome type. Build the model that performs with the accuracy better than the baseline - 70%

## Project's Description
Austin Animal Center is the municipal shelter for the City of Austin and unincorporated Travis County. Together with Austin Pets Alive! this shelter is doing an amazing job on keeping as many animals as possible away from  euthanasia. Even animals that entered the shelter with an Euthanasia Request get euthanized only in less than 25% of cases.
Data  contains records from October 1, 2013 till December 12, 2022 included. During this time period almost 100K animals went through the shelter. Around 70% of them were adopted, 2.5% euthanized. This is the lowest rate across United States. Creating the classification model potentially can help identify the animals at the high risk of euthanasia and help to put more efforts for the happy ending for those animals.

## Executive Summary
Quick data from the project:
- This is a classification project.
- The data merged from two data sets: Intake and Outcomes. Both data sets contain the information about same animals and they were merged based on the animal ID.
- 70% of animals are getting adopted, 27% transfered to other facilities, 2.5% euthinized and less than 1% dies of natural causes.
- Exploration analysis showed that most of the features are significant for the outcome predictions.
- My best performing model has the accuracy 80.3%

### Data source
Data was obtained from the austintexas.gov website on December 12, 2022. 
Links for the source:
- [Intakes](https://data.austintexas.gov/Health-and-Community-Services/Austin-Animal-Center-Intakes/wter-evkm)

- [Outcomes](https://data.austintexas.gov/Health-and-Community-Services/Austin-Animal-Center-Outcomes/9t4d-g238)

You can also find a cleaned data set in ```austin_animal_shelter.csv``` file.


## Steps to Reproduce
1) Clone this repo into your computer.
4) Run the ```austin_animal_center.ipynb``` file.

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
| Feature | Definition or Unique Values | Manipulations applied|Data Type|
|--------|-----------|-----------|-----------|
||
|||**Identification Data**
||
|*animal_id*| Registration number of the animal  | -| object
|*name*|Name of the animal|-|object
|||**Categorical Data**
||
|*animal_type*|'Cat', 'Dog', 'Rabbit', 'Wild', 'Bird', 'Guinea Pig', 'Livestock'| Replaces 'Other' with the type based on ```breed```| category
|*intake_type*|'Stray', 'Owner Surrender', 'Public Assist', 'Euthanasia Request', 'Wildlife', 'Abandoned'|-|category
|*intake_condition*|'Normal', 'Injured', 'Sick' etc. 14 different values| - | category
|*sex_of_animal*| 'Male', 'Female', 'Unknown' | Derived from ```sex``` column| category
|*sterilized*| 'Spayed', 'Neutered', 'Intact', 'Unknown' | Derived from ```sex``` column| category
|*sex*| 'Spayed Female', 'Neutered Male', 'Intact Male', 'Intact Female', 'Unknown' | |integer
|*breed*| breeds of cats and dogs | All other animals were replaced with 'Unknown' and moved to ```animal_type```. | category
|*color*| Color of the animal |  | category
|*outcome_subtype*| 'Foster', 'Emergency', 'Partner', 'Court/Investigation', 'Rabies Risk' etc | | category
|*mixed_breed*, *domestic_breed*, *pitbull*| 0/1 Dummy variables for modeling | | uint8
||
|||**Date types**
|*first_check_in*| The date when the animal first checked in at the shelter |Minimum value for the ```animal_id``` from ```DateTime_in``` column of the original data set|datetime
|*last_check_out*| The last check out date of the animal |Maximum value for the ```animal_id``` from ```DateTime_out``` column of the original data set |datetime
|*date_of_birth*| Animal's date of birth | | datetime
|||**Numerical Values**
|*age_on_check_in*| Age in days when the animal first checked in |```first_check_in``` - ```date_of_birth```|integer
|*times_in_shelter*| How many times the animal was checked in at the shelter | Counted how many times ```animal_id``` appeared in the original data set |integer
|*days_at_shelter*| How many days the animal is in the shelter | ```last_check_out``` - ```first_check_in``` | integer
|*age_in_days*, *age_in_months*, *age_in_years*| Age of animal in days, months, years | ```last_check_out``` - ```date_of_birth```| datetime
|||**Target Data**
||
|**outcome_type** | **'Adoption', 'Transfer', 'Euthanasia', 'Died'** |'Return to Owner', 'RTO/Adopt' -> merged with 'Adoption', 'Relocate' -> merged with 'Transfer', 'Missing', 'Stolen', 'Lost' -> dropped, just few values| **float**

#### Data preparation takeaways:
To clean data:
- Made every row = 1 observation
- Removed the columns that are presented in both ```intake``` and ```outcome``` data sets
- Created new features that describe age of the animal, how long it stayed at the shelter, how many times checked in
- Removed some of the outliers:
 * As Austin shelter has the program which allows people to take an animal for the weekend/holiday or just for a walk, some animals had up to 1800+ check ins. Those cases are rare. Seven animals that checked in more than 20 times were removed.

**Final results**
The original merged data contained 188446 rows and 23 columns.
The cleaned data has 97799 rows and 24 columns

#### Exploration Questions and Takeaways
1. What percentage of animals is adopted, transfered, euthanized or dies of natural conditions?
 - Almost 70% of animals are adopted, around 26% are transfered to other facilities, 2.5% of animals are euthanized and less than 1% die of natural causes.
 2. Is the animal type related to the outcome?
 - The outcome type has an association with the animal type. Animals other than cats, dogs, or rabbits get euthanized much more often, and cats/dogs are euthanized more often than rabbits.
 3. Does the sex of the animal have a relation to the outcome?
 - Intact animals have much lower chances to be adopted.
 - Despite the distribution of outcomes for males and females look relatively the same, the statistical test shows that the sex plays a significant role in the outcome. It might happen because of 'Unknown' sex group.
 4. Is the age of the animal connected to the outcome type?
 - Older animals have higher chances to be euthanized while very young animals die more from natural causes.
 - There is a significant difference in means between the age on the check-in and overall mean among different outcomes.
 - There is a significant difference in means of the ```age_on_check_in``` among all outcome groups.
 5. How differ the average age of the animal from its type? And how does it affect the outcome type?
 - Beeing an old cat or an old guinea pig dramatically increase the chance of the euthanasia for an animal. While wild animals have almost equal chances to be adopted, transfered, euthanized or die not depending on their age.

 #### Modeling summary
 - It is possible to make the prediction model that performs well using the features only from the ```Austin_Animal_Center_Intakes.csv``` data set.
- The best performing model has the accuracy 80.3%

### Conclusion and next steps
The goals of this project were:
- Find the features that will help us to predict our target variable - outcome type. 
- Build the model that performs with the accuracy better than the baseline - 70%

My model performs 10% better than the baseline model. I created the additional features and they helped to improve the model performance. On the `intake only` data the model with engineered features performs 0.05% better than the model without them.

As the next step for this project I would try to build the regression model that predicts how many day the animal will stay in the shelter. In could help with the financial planning.