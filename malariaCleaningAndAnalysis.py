#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'Data-Analysis-Project'))
	print(os.getcwd())
except:
	pass

#%%
#import libs
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
import numpy.random as nr
import math

#%%
# load datasets
Malaria = pd.read_csv('./NG_2015_MIS_07012019_1354_135943/nmis.csv')

#%%
Malaria.head(20)

#%%
# assign human-readable names to column/variable names in the dataset

#%%
Malaria.columns=['Case Identification', 'Region', 'Type of Place of Residence', 'Source of Drinking Water', 'Type of Toilet Facility',
                'Has Electricity', 'Main Floor Material', 'Main Wall Material', 'Main Roof Material', 'Has Bicycle', 'Has Motorcycle/Scooter',
                'Has Car/Truck', 'Has Mosquito Bed Net for Sleeping', 'Owns Land Suitable for Agriculture', 'Has Bank Account', 
                'Wealth Index', 'Cost of Treatment for Fever', 'State']

print(Malaria.shape)

#%%
Malaria.head()

#%%
# Some of the column/variable names contain wild/special chars
Malaria.columns=[str.replace('/','or') for str in Malaria.columns]

#%%
Malaria.head()

#%%
Malaria.dtypes

# We are going to check which of our variables have missing values

#%%
# check for missing values
Malaria.isnull().sum(axis=0) 

#%% [markdown]
#  'Type of Toilet Facility' and 'Cost of Treatment for Fever' contains missing values. We are going to remove both columns from our dataset.

#%%
Malaria.drop(['Cost of Treatment for Fever','Type of Toilet Facility'], axis=1, inplace=True)


#%%
Malaria.head()

#%%
#put our table in form of pandas dataframe for analysis
df = pd.DataFrame(Malaria)

#%%
df

#%%
#Descriptive statistics/analysis

#%%
df['State'].value_counts()


#%%
df['Has Electricity'].value_counts()


#%%
df['Source of Drinking Water'].value_counts()


#%%
df['Wealth Index'].value_counts()


#%%
df['Has Mosquito Bed Net for Sleeping'].value_counts()


#%%
df.groupby('Wealth Index')['State'].describe()

#%% [markdown]
# From the table above, Lagos State has the top number of richest people and Sokoto State has the top number of poorest people.

#%%
df.groupby('Has Mosquito Bed Net for Sleeping')['State'].describe()

#%% [markdown]
# From above table Bauchi State has the Highest number of people with Mosquito Bed Net for Sleeping, While Edo State has the least Number.

#%%
df.groupby('Has Electricity')['State'].describe()

#%% [markdown]
# From above table Lagos State has the Highest number of people with access to Electricity, While Adamawa State has the least Number.

#%%
df['Source of Drinking Water'].value_counts()

#%% [markdown]
# From the table above, most people source of drinking water is Tube Well or Borehole.

#%%
