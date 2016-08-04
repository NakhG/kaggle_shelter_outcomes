

#INITIAL STEPS

#Import fundamental libraries/packages
import numpy as np
import pandas as pd

#Change working directory to where the data is
import os
os.chdir('C:\\Users\\gnakhleh\\Documents\\Self Guided\\kaggle_shelter')

#Read in the data
shelter_df = pd.read_csv('train.csv')



#DATA CLEANING/PREP

#Create new variables

#Convert Age to days
Age = shelter_df.AgeuponOutcome
#split the strings into two, expand = True making the result a dataframe
Age_split_df = Age.str.split(' ', expand=True)
#merge the original df w/ this new one and rename the new columns
shelter_df = pd.concat([shelter_df, Age_split_df], axis=1)
shelter_df = shelter_df.rename(columns={0:'AgeNum', 1: 'TimePeriod'})
#set AgeNum to be numeric
shelter_df['AgeNum'] = shelter_df['AgeNum'].convert_objects(convert_numeric=True)
#looping over rows is less efficient than working w/ numpy
shelter_df['AgeNum'][(shelter_df['TimePeriod'] == 'week') | (shelter_df['TimePeriod'] == 'weeks')] = shelter_df['AgeNum'] * 7
shelter_df.head()
#change for months
shelter_df['AgeNum'][(shelter_df['TimePeriod'] == 'month') | (shelter_df['TimePeriod'] == 'months')] = shelter_df['AgeNum'] * 30
shelter_df['AgeNum'][(shelter_df['TimePeriod'] == 'month') | (shelter_df['TimePeriod'] == 'months')].head()
#change for years
shelter_df['AgeNum'][(shelter_df['TimePeriod'] == 'year') | (shelter_df['TimePeriod'] == 'years')] = shelter_df['AgeNum'] * 365

#OK, we now have a dataframe w/ age upon outcome (in days)
#Lets rename AgeNum to AgeDays
shelter_df = shelter_df.rename(columns={'AgeNum':'AgeDays'})


#Create new column for Mix? (Y/N)
'''
How: loop over the Breed column, looking for "/"'s or the word 'mix'
...if it finds those, append a "1" to the blank list that will become our new column
...if not, give a zero
'''
mix = []
for row in shelter_df['Breed']:
    if 'mix' in row.lower() or '/' in row.lower():
        mix.append(1)
    else:
        mix.append(0)
#Create the new Mix column from the list we made        
shelter_df['Mix?'] = mix    

#Create new column for Primary Color
'''
How: loop  over the Color column, looking for "/"'s
...if found, split and take the first entry, if not, take what is found
'''
prime_color = []
for row in shelter_df['Color']:
    if '/' in row:
        colorsplit = row.split('/')
        prime_color.append(colorsplit[0])
    else:
        prime_color.append(row)
#Create the new PrimaryColor column
shelter_df['PrimaryColor'] = prime_color


#Creating a smaller dataframe to drop na's and use for analysis

#Make a new dataframe, the combination of the predictors and outcome, with NaN's dropped
Xy_noNan = shelter_df[['OutcomeType', 'AnimalType', 'SexuponOutcome', 'AgeDays', 'Mix?', 'PrimaryColor']].dropna(axis=0)
#Encode string entries as numbers using the preprocessing module from sklearn 
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
Xy_noNan['AnimalType_float'] = le.fit_transform(Xy_noNan['AnimalType'])
Xy_noNan['SexuponOutcome_float'] = le.fit_transform(Xy_noNan['SexuponOutcome'])
Xy_noNan['PrimaryColor_float'] = le.fit_transform(Xy_noNan['PrimaryColor'])


#Split the dataframe Xy_noNan into training and testing sets

#First, we split Xy_noNan into X and y dataframes
X = Xy_noNan[['AnimalType_float','SexuponOutcome_float', 'PrimaryColor_float', 'AgeDays', 'Mix?']]
y = Xy_noNan['OutcomeType']

#Next, use train_test_split function from sklearn, doing a stratified sample to preserve the outcome breakdown
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=39, stratify = y)
#We now have our training and testing datasets, split into predictors and outcome for each set



#DATA MODELING

#Model 1: Decision tree
#Import the decision tree module from sklearn
from sklearn import tree

#Fit the tree to the training data
clf = tree.DecisionTreeClassifier(max_depth=10, criterion="entropy")
clf.fit(X_train, y_train)
c = clf.fit(X_train, y_train)

#Test how our model did
#Use the confusion matrix function from sklearn
from sklearn.metrics import confusion_matrix
y_true = y_train
y_pred = c.predict(X_train)

confusion_matrix(y_true, y_pred) #tough to read, but the diagonal is showing us our accurate predictions, all off-diag are wrong

#Simpler view
c.score(X_test,y_test) #62% accuracy. Not great!

#Why is this decision tree inaccurate? 
#Look at the decision tree graphic to see what rubric is failing
'''
FOR LATER: 
should we make new variables??
what different models can we make (logistic regression 'ensemble'?)
'''

#fitting a multiclass logistic regression

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
log_model = logreg.fit(X_train, y_train)
log_model.score(X_train, y_train)  #0.54, worse than the decision tree

#optimizing a decision tree w/ gradient boosting classifier
from sklearn.ensemble import GradientBoostingClassifier
boost_model = GradientBoostingClassifier(n_estimators=50).fit(X_train, y_train)
boost_model.score(X_test, y_test)

#try Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
rforest_model = RandomForestClassifier(n_estimators=70).fit(X_train, y_train)
rforest_model.score(X_test, y_test)


'''
Bringing in new data:
Reading in extra data the Austin shelter has on their website
We'll join on AnimalID and bring in new variables, and re-do our cleaning/prep
'''

intake_data_1415 = pd.read_csv("intake_data_combined.csv", encoding='latin-1')

#Lets try a left join

shelter_df_merged = pd.merge(shelter_df, intake_data_1415, how='left', on=['AnimalID'])

shelter_df.columns
intake_data_1415.columns  #our problem is that AnimalID is called 'Animal ID' in the intake file

#rename the Animal ID column in the intake dataset

intake_data_1415.rename(columns={'Animal ID' : 'AnimalID'}, inplace=True)

shelter_df_merged = pd.merge(shelter_df, intake_data_1415, how='left', on=['AnimalID'])

#Some EDA on our new merged dataset
shelter_df_merged.head()

pd.crosstab(shelter_df_merged['Intake Type'], shelter_df_merged['OutcomeType'])
