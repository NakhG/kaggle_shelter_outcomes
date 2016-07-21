# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 11:12:21 2016

@author: GNAKHLEH
"""

import numpy as np
import pandas as pd

import os
os.chdir('C:\\Users\\gnakhleh\\Documents\\Self Guided\\kaggle_shelter')

shelter_df = pd.read_csv('train.csv')

shelter_df.head()

#a list of all unique outcome types and outcome subtypes

shelter_df['OutcomeType'].unique() , shelter_df['OutcomeSubtype'].unique()

#get column names
list(shelter_df)

'''
Multi-class classification: what are our options?
One-vs-rest or One-vs-one
Specific examples:
Decision trees / random forests, naive bayes, LDA and QDA, Nearest Neighbors
'''

#Let's try a decision tree

#Bring in the decision tree module from scikit learn
from sklearn import tree

#Get the predictor variables together
X = shelter_df[['AnimalType', 'SexuponOutcome', 'AgeuponOutcome', 'Breed', 'Color']]
y = shelter_df['OutcomeType']

X.head()

#let's standardize those ages
#do it in the original df, since we might need it for other reasons
shelter_df.AgeuponOutcome.unique()



'''
Lets split the entries: numbers and time grouping (eg days, months, years)
That way, we can do standardization in a programmatic way
Ex: if == weeks, multiply number by 7
Outcome: all in days format
'''

Age = shelter_df.AgeuponOutcome
#split the strings into two, expand = True making the result a dataframe
Age_split_df = Age.str.split(' ', expand=True)

#merge the original df w/ this new one and rename the new columns
shelter_df = pd.concat([shelter_df, Age_split_df], axis=1)
shelter_df = shelter_df.rename(columns={0:'AgeNum', 1: 'TimePeriod'})

shelter_df.head()

#reorder the columns so it makes more sense
list(shelter_df)
shelter_df = shelter_df[['AnimalID', 'Name', 'DateTime', 
'OutcomeType', 'OutcomeSubtype','AnimalType',
'SexuponOutcome','AgeuponOutcome','AgeNum', 'TimePeriod','Breed','Color']]
shelter_df.head()

shelter_df.describe()
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

#lets update the X for our models
X = shelter_df[['AnimalType', 'SexuponOutcome', 'AgeDays', 'Breed', 'Color']]

X.head()


#OK so we have our X and Y
#let's just try to run a decision tree on it

clf = tree.DecisionTreeClassifier(max_depth=10, criterion="entropy")
clf.fit(X, y)   #something wrong: wanted to convert color to a float. why?

#lets look at some info on that Color column
table = pd.crosstab(index=X['Color'], columns="count") #there are over 350 different colors present in the data, many w/ counts less than 10
#should we bin this data?

#TAKE A STEP BACK: let's look at each variable a bit and determine any changes we want to make
#We know there are WAY too many colors and breeds to work w/
#so we should make some variables out of at least those two

#I want to make Two new columns:
#Mix: using species, if there's a 'Mix', 1, if not 0
#Primary color: using color, split on the '/' and return the 1st entry in that list

#create the two columns

mix = []

for row in shelter_df['Breed']:
    if 'mix' in row.lower() or '/' in row.lower():
        mix.append(1)
    else:
        mix.append(0)
        
shelter_df['Mix?'] = mix    

prime_color = []
for row in shelter_df['Color']:
    if '/' in row:
        colorsplit = row.split('/')
        prime_color.append(colorsplit[0])
    else:
        prime_color.append(row)

shelter_df['PrimaryColor'] = prime_color



#Retry making a decision tree model
X = shelter_df[['AnimalType', 'SexuponOutcome', 'AgeDays', 'Mix?', 'PrimaryColor']]
y = shelter_df['OutcomeType']
clf = tree.DecisionTreeClassifier(max_depth=10, criterion="entropy")
clf.fit(X, y)

#Still doesn't work
#Why? Python .fit() methods prefer floats, won't work w/ strings
#There's also a problem w/ working w/ NaN here: maybe we can start by only working w/ rows that have no NaN's

Xy_noNan = shelter_df[['OutcomeType', 'AnimalType', 'SexuponOutcome', 'AgeDays', 'Mix?', 'PrimaryColor']].dropna(axis=0)
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
Xy_noNan['AnimalType_float'] = le.fit_transform(Xy_noNan['AnimalType'])
Xy_noNan['SexuponOutcome_float'] = le.fit_transform(Xy_noNan['SexuponOutcome'])
Xy_noNan['PrimaryColor_float'] = le.fit_transform(Xy_noNan['PrimaryColor'])

X = Xy_noNan[['AnimalType_float','SexuponOutcome_float', 'PrimaryColor_float', 'AgeDays', 'Mix?']]
y = Xy_noNan['OutcomeType']
clf.fit(X, y)

#ok the decision tree was fit
#lets look at it

from sklearn.externals.six import StringIO
with open("dtree_Shelter1.dot", 'w') as f:
    f = tree.export_graphviz(clf, out_file = f)


#
#import pydot