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
shelter_df['AgeNum'][(shelter_df['TimePeriod'] == 'weeks') | (shelter_df['TimePeriod'] == 'week')] = shelter_df['AgeNum']*7

shelter_df['AgeNum'][(shelter_df['TimePeriod'] == 'weeks') | (shelter_df['TimePeriod'] == 'week')].head()

shelter_df.head()