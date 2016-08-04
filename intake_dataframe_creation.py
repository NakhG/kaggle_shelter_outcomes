# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 16:55:16 2016

@author: gnakhleh
"""

#Kaggle Shelter Outcomes

'''
This script reads in outside data from the Austin shelter website
Link: http://www.austintexas.gov/department/reports-1
Two files: intakes from FY 2014 and FY 2015

Goal here is to read in the two files, append them together, forming one dataset
Export that as a csv

Then we will go back to our main script, read this data in
Then we will match the AnimalID and join some variables over
'''

import numpy as np
import pandas as pd

import os

os.chdir('C:\\Users\\gnakhleh\\Documents\\Self Guided\\kaggle_shelter')

#read in the datasets
intake_data_14 = pd.read_csv("Austin_Animal_Center_FY14_Intakes.csv")

intake_data_15 = pd.read_csv("Austin_Animal_Center_FY15_Intakes__Updated_Hourly_.csv")

#append 2015 data to 2014

intake_data_1415 = intake_data_14.append(intake_data_15, ignore_index=True)


#Test: grab a random record from the 2015 dataset, and see if it matches where it should be in the new combined dataframe
intake_data_15.shape   #18627

intake_data_15.iloc[18500:18501, :]   #AnimalID = A712786

intake_data_14.shape  #18729 rows

intake_data_1415.shape # 37356 rows

18729 + 18500   #should be around 37229'th row

intake_data_1415.iloc[37224:37230, :]   #checks out

#export the combined dataset

intake_data_1415.to_csv('intake_data_combined.csv')