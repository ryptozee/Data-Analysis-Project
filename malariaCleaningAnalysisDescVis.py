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
#APPLICATION OF MACHINE LEARNING MODELS

#%%
Malaria.head()

#%%
Malaria_ML = pd.read_csv("./NG_2015_MIS_07012019_1354_135943/numeric_nmis.csv")

#%%
Malaria_ML.head()

#%%
Malaria_ML.columns=['Case Identification', 'Region', 'Type of Place of Residence', 'Source of Drinking Water', 'Type of Toilet Facility',
                'Has Electricity', 'Main Floor Material', 'Main Wall Material', 'Main Roof Material', 'Has Bicycle', 'Has Motorcycle/Scooter',
                'Has Car/Truck', 'Has Mosquito Bed Net for Sleeping', 'Owns Land Suitable for Agriculture', 'Has Bank Account', 
                'Wealth Index', 'Cost of Treatment for Fever', 'State']

print(Malaria_ML.shape)
Malaria_ML.head()

#%%
Malaria_ML.columns=[str.replace('/','or') for str in Malaria_ML.columns]


#%%
Malaria_ML.isnull().sum(axis=0) 

#%%
Malaria_ML.drop(['Cost of Treatment for Fever','Type of Toilet Facility'], axis=1, inplace=True)

#%%
Malaria_ML.head()

#%%
Malaria_ML.tail()

#%%
def plot_corr(Malaria_ML, size=11):
    """
    Function plots a graphical correlation matrix for each pair of columns in the dataframe.

    Input:
        Malaria: pandas DataFrame
        size: vertical and horizontal size of the plot

    Displays:
        matrix of correlation between columns.  Blue-cyan-yellow-red-darkred => less to more correlated
                                                0 ------------------>  1
                                                Expect a darkred line running from top left to bottom right
    """

    corr = Malaria_ML.corr()    # data frame correlation function
    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(corr)   # color code the rectangles by correlation value
    plt.xticks(range(len(corr.columns)), corr.columns)
    # draw x tick marks
    plt.yticks(range(len(corr.columns)), corr.columns)
    # draw y tick marks

#%%
# Correlated Feature Check. 
# Correlation by color. Red is most correlated with other variable, Yellow is self to self correlated and Blue is least correlated with other variable.

#%%
plot_corr(Malaria_ML)

#%%
# State and Case Identification appears to be correlated.
# Drop State Column

del Malaria_ML['State']

#%%
Malaria_ML.head(5)


#%%
Malaria_ML.corr()

#%%
plot_corr(Malaria_ML)

#%%
# The correlations look good. There appear to be no coorelated columns.
# Next we want to check class distribution
#%%
num_obs = len(Malaria_ML)
num_true = len(Malaria_ML.loc[Malaria_ML['Has Mosquito Bed Net for Sleeping'] == 1])
num_false = len(Malaria_ML.loc[Malaria_ML['Has Mosquito Bed Net for Sleeping'] == 0])
print("Number of True cases: {0} ({1:2.2f}%)".format(num_true, (num_true/num_obs) * 100))
print("Number of False cases: {0} ({1:2.2f}%)".format(num_false, (num_false/num_obs) * 100))

#%% [markdown]
# Our class distribution is fairly good.

#%% [markdown]
# # Spliting the data
# 70% for training, 30% for testing

#%%
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
feature_col_names = ['Region', 'Type of Place of Residence', 'Source of Drinking Water', 'Has Electricity', 'Wealth Index', 'Has Bicycle', 'Has MotorcycleorScooter', 'Has CarorTruck' , 'Owns Land Suitable for Agriculture', 'Has Bank Account' , 'Main Floor Material' ,'Main Wall Material' , 'Main Roof Material']
predicted_class_names = ['Has Mosquito Bed Net for Sleeping']

X = Malaria_ML[feature_col_names].values     # predictor feature columns (8 X m)
y = Malaria_ML[predicted_class_names].values # predicted class (1=true, 0=false) column (1 X m)
split_test_size = 0.30

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_test_size, random_state=42) 
# test_size = 0.3 is 30%, 42 is the answer to everything

#%%
#Get an idea bout the rows and columns we have obtained
print("\nX_train:\n")
print(X_train.shape)

print("\nX_test:\n")
print(X_test.shape)
#%%
# check we have the the desired 70% train, 30% test split of the data.

#%%
print("{0:0.2f}% in training set".format((len(X_train)/len(Malaria_ML.index)) * 100))
print("{0:0.2f}% in test set".format((len(X_test)/len(Malaria_ML.index)) * 100))

#%%
# Verifying predicted value was split correctly.

#%%
print("Original True  : {0} ({1:0.2f}%)".format(len(Malaria_ML.loc[Malaria_ML['Has Mosquito Bed Net for Sleeping'] == 1]), (len(Malaria_ML.loc[Malaria_ML['Has Mosquito Bed Net for Sleeping'] == 1])/len(Malaria_ML.index)) * 100.0))
print("Original False : {0} ({1:0.2f}%)".format(len(Malaria_ML.loc[Malaria_ML['Has Mosquito Bed Net for Sleeping'] == 0]), (len(Malaria_ML.loc[Malaria_ML['Has Mosquito Bed Net for Sleeping'] == 0])/len(Malaria_ML.index)) * 100.0))
print("")
print("Training True  : {0} ({1:0.2f}%)".format(len(y_train[y_train[:] == 1]), (len(y_train[y_train[:] == 1])/len(y_train) * 100.0)))
print("Training False : {0} ({1:0.2f}%)".format(len(y_train[y_train[:] == 0]), (len(y_train[y_train[:] == 0])/len(y_train) * 100.0)))
print("")
print("Test True      : {0} ({1:0.2f}%)".format(len(y_test[y_test[:] == 1]), (len(y_test[y_test[:] == 1])/len(y_test) * 100.0)))
print("Test False     : {0} ({1:0.2f}%)".format(len(y_test[y_test[:] == 0]), (len(y_test[y_test[:] == 0])/len(y_test) * 100.0)))

#%%
# # Training Algorithm - Using Naive Bayes Machine Learning Model
# # Using Logistic Regression
from sklearn.naive_bayes import GaussianNB

# create Gaussian Naive Bayes model object and train it with the data
nb_model = GaussianNB()

nb_model.fit(X_train, y_train.ravel())

#%% [markdown]
# Performance on Training Data

#%%
# predict values using the training data
nb_predict_train = nb_model.predict(X_train)

# import the performance metrics library
from sklearn import metrics

# Accuracy
print("Accuracy: {0:.0f}%".format(metrics.accuracy_score(y_train, nb_predict_train)*100))
print()

#%% [markdown]
# Our accurancy rate is 63% on the training data. This is below the 70% benchmark for our ideal ML Model.
#%% [markdown]
# Performance on Testing Data

#%%
# predict values using the testing data
nb_predict_test = nb_model.predict(X_test)

from sklearn import metrics

# training metrics
print("nb_predict_test", nb_predict_test)
print ("y_test", y_test)
print("Accuracy: {0:.0f}%".format(metrics.accuracy_score(y_test, nb_predict_test)*100))


#%%
#Accuracy on testing data is also below our 70% benchmark.


#%%
print("Confusion Matrix")
print("{0}".format(metrics.confusion_matrix(y_test, nb_predict_test)))
print("")

print("Classification Report")
print(metrics.classification_report(y_test, nb_predict_test))

#%% [markdown]
# Our Recall and Precision rate is 70% and 77% respectively. This is ok. However we would try other models if they would work better.
#%% [markdown]
# # Using Random Forest

#%%
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(random_state=42, n_estimators=10)      # Create random forest object
rf_model.fit(X_train, y_train.ravel()) 

#%%
# Predict Training Data

#%%
rf_predict_train = rf_model.predict(X_train)
# training metrics
print("Accuracy: {0:.0f}%".format(metrics.accuracy_score(y_train, rf_predict_train)*100))

#%% [markdown]
# Random Forest Accuracy level looks much better.
#%% [markdown]
# Predict Test Data

#%%
rf_predict_test = rf_model.predict(X_test)

# training metrics
print("Accuracy: {0:.0f}%".format(metrics.accuracy_score(y_test, rf_predict_test)*100))

#%% [markdown]
# But this is slightly below 70% for our test data.

#%%
print(metrics.confusion_matrix(y_test, rf_predict_test) )
print("")
print("Classification Report")
print(metrics.classification_report(y_test, rf_predict_test))

#%% [markdown]
# Our precision and Recall recorded good values based on true 'Yes' and 'No' for ownership of Mosquito Bed Net for Sleeping though the accuracy level on the test data is slightly less than our 70% benchmark.
#%% [markdown]
# # Using Logistic Regression

#%%
from sklearn.linear_model import LogisticRegression

lr_model =LogisticRegression(C=0.7, random_state=42, solver='liblinear', max_iter=10000)
lr_model.fit(X_train, y_train.ravel())
lr_predict_test = lr_model.predict(X_test)

# training metrics
print("Accuracy: {0:.0f}%".format(metrics.accuracy_score(y_test, lr_predict_test)*100))
print("Precision: {0:.0f}%".format(metrics.precision_score(y_test, lr_predict_test)*100))
print("Recall: {0:.0f}%".format(metrics.recall_score(y_test, lr_predict_test)*100))
print("")
print("Classification Report")
print(metrics.classification_report(y_test, lr_predict_test))

#%% [markdown]
# Logistic Regression Model performed best for our prediction. So we would finally go with the Logistics Regression Model.
#%% [markdown]
# # Using our trained Model (Logistic Regression)
#%% [/]
# Save trained model to file
from sklearn.externals import joblib  
joblib.dump(lr_model, "Malaria Model")

#%%
#load trained model
lr_model = joblib.load('Malaria Model')

#%%
#Test prediction on data and once the model is loaded

Malaria_Predic = pd.read_csv("./NG_2015_MIS_07012019_1354_135943/numeric_mtd.csv")

#%%
Malaria_Predic.head()

#%%
#Test data contains a few rows

#%%
#We will do some cleaning as before
Malaria_Predic.columns=['Case Identification', 'Region', 'Type of Place of Residence', 'Source of Drinking Water', 'Type of Toilet Facility',
                'Has Electricity', 'Main Floor Material', 'Main Wall Material', 'Main Roof Material', 'Has Bicycle', 'Has Motorcycle/Scooter',
                'Has Car/Truck', 'Has Mosquito Bed Net for Sleeping', 'Owns Land Suitable for Agriculture', 'Has Bank Account', 
                'Wealth Index', 'Cost of Treatment for Fever', 'State']
print(Malaria_Predic.shape)
Malaria_Predic.head()

#%%
Malaria_Predic.columns=[str.replace('/','or') for str in Malaria_Predic.columns]

#%%
Malaria_Predic.drop(['Type of Toilet Facility', 'Cost of Treatment for Fever', 'Case Identification', 'State'], axis=1, inplace=True)


#%%
Malaria_Predic.head()

#%%
#We need to drop 'Has Mosquito Bed Net for Sleeping" since that is what we are preicting
#Store data without the column with prefix X as we did with the X_train and X_test to indicate that it only contains the columns we are predicting

#%%
X_predic = Malaria_Predic
del X_predic['Has Mosquito Bed Net for Sleeping']

#%%
X_predic

#%% [markdown]
# At this point our data is ready to be used for prediction.

#%% [markdown]
# Predict 'Has Mosquito Bed Net for Sleeping' with the prediction data. Returns 1 if True, 0 if false

#%%
Malaria_Predic.head()

#%%
lr_model.predict(X_predic)

#%%
# Our Model predicts well. Mision Accomplished!!

Malaria_Visual = pd.read_csv("./NG_2015_MIS_07012019_1354_135943/numeric_nmis.csv")
Malaria_Visual.columns=['Case Identification', 'Region', 'Type of Place of Residence', 'Source of Drinking Water', 'Type of Toilet Facility',
                'Has Electricity', 'Main Floor Material', 'Main Wall Material', 'Main Roof Material', 'Has Bicycle', 'Has Motorcycle/Scooter',
                'Has Car/Truck', 'Has Mosquito Bed Net for Sleeping', 'Owns Land Suitable for Agriculture', 'Has Bank Account', 
                'Wealth Index', 'Cost of Treatment for Fever', 'State']

print(Malaria_Visual.shape)
Malaria_Visual.head()

#%%
#Check for Missing Values
(Malaria_Visual.astype(np.object).isnull()).any()

#%% [markdown]
# Column 'Cost of Treatment of Fever' containing NaN values is removed.

#%%
Malaria_Visual.drop('Cost of Treatment for Fever', axis = 1, inplace = True)


#%%
Malaria_Visual.head()

#%% [markdown]
# We would put the table in form of Pandas DataFrame.

#%%
df=pd.DataFrame (Malaria_Visual)

#%% [markdown]
# Now we would create and assign a list of dictionaries to recode the numerical values of SOME categorical variables in our dataset with human-readable text.

#%%
dict = [['Has Electricity',
        {1:'yes',
        0:'No'}],
       ['Type of Place of Residence',
       {1:'Urban',
       2:'Rural'}]]
for col_dict in dict:
    col=col_dict[0]
    dict=col_dict[1]
    df[col]=[dict[x] for x in df[col]]


#%%
dict = [['Source of Drinking Water',
        {10:'Piped water',
         11:'Piped into dwelling',
         12:'Piped to yard/plot',
         13:'public tap/standpipe',
         14:'Piped to Neighbour',
         20:'Tube well water',
         21:'Tube well or borehole',
         30:'Dug well (open/protected)',
         31:'Protected well',
         32:'Unprotected well',
         40:'Surface water',
         41:'Protected spring',
         42:'Unprotected spring',
         43:'River/dam/lake/ponds/stream/canal/irrigation channel',
         51:'Rain water',
         61:'Tanker truck',
         62:'Cart with small tank',
         71:'Bottled water',
         72:'Sachet water',
         96:'Other'}]]

for col_dict in dict:
    col=col_dict[0]
    dict=col_dict[1]
    df[col]=[dict[x] for x in df[col]]


#%%
dict = [['Region',
       {1:'North central',
        2:'North east',
        3:'North west',
        4:'South east',
        5:'South south',
        6:'South west'}]]
for col_dict in dict:
    col=col_dict[0]
    dict=col_dict[1]
    df[col]=[dict[x] for x in df[col]]


#%%
dict = [['State',
        {10:'Sokoto',
        20:'Zamfara',
        30:'Katsina',
        40:'Jigawa',
        50:'Yobe',
        60:'Borno-Urban',
        70:'Adamawa',
        80:'Gombe',
        90:'Bauchi',
        100:'Kano',
        110:'Kaduna',
        120:'Kebbi',
        130:'Niger',
        140:'FCT Abuja',
        150:'Nasarawa',
        160:'Plateau',
        170:'Taraba',
        180:'Benue',
        190:'Kogi',
        200:'Kwara',
        210:'Oyo',
        220:'Osun',
        230:'Ekiti',
        240:'Ondo',
        250:'Edo',
        260:'Anambra',
        270:'Enugu',
        280:'Ebonyi',
        290:'Cross River',
        300:'Akwa Ibom',
        310:'Abia',
        320:'Imo',
        330:'Rivers',
        340:'Bayelsa',
        350:'Delta',
        360:'Lagos',
        370:'Ogun'}]]

for col_dict in dict:
    col=col_dict[0]
    dict=col_dict[1]
    df[col]=[dict[x] for x in df[col]]


#%%
dict = [['Has Bank Account',
        {1:'yes',
         0: 'No'}], 
         ['Has Bicycle',
         {1:'yes',
         0:'No'}]]         
        
for col_dict in dict:
    col=col_dict[0]
    dict=col_dict[1]
    df[col]=[dict[x] for x in df[col]]


#%%
dict = [['Has Mosquito Bed Net for Sleeping',
        {1:'yes',
         0: 'No'}], 
         ['Has Car/Truck',
         {1:'yes',
         0:'No'}]]         
        
for col_dict in dict:
    col=col_dict[0]
    dict=col_dict[1]
    df[col]=[dict[x] for x in df[col]]


#%%
dict = [['Wealth Index',
        {1:'Poorest',
         2:'Poorer',
         3:'Middle',
         4:'Richer',
         5:'Richest'}]]         
        
for col_dict in dict:
    col=col_dict[0]
    dict=col_dict[1]
    df[col]=[dict[x] for x in df[col]]


#%%
dict = [['Has Motorcycle/Scooter',
        {1:'yes',
         0: 'No'}]]         
        
for col_dict in dict:
    col=col_dict[0]
    dict=col_dict[1]
    df[col]=[dict[x] for x in df[col]]


#%%
df

#%% [markdown]
# Fine with the missing values check and recoding of some categorical variables
# Now on to visualizing the dataset.

#%%
def plot_box(df, cols, col_x = 'Has Mosquito Bed Net for Sleeping'):
    for col in cols:
        sb.set_style("whitegrid")
        sb.boxplot(col_x, col, data=df)
        plt.xlabel(col_x) # Set text for the x axis
        plt.ylabel(col)# Set text for y axis
        plt.show()

num_cols = ['Case Identification']
plot_box(df, num_cols)

#%% [markdown]
# From the boxplot above, there is obvious gap in the number of people who indicated having no Mosquito Bed Net for Sleeping and those who indicated they have. 

#%%
def plot_box(df, col, col_y = 'Case Identification'):
    sb.set_style("whitegrid")
    sb.boxplot(col, col_y, data=df)
    plt.xlabel(col) # Set text for the x axis
    plt.ylabel(col_y)# Set text for y axis
    plt.show()
    
plot_box(df, 'Wealth Index')    

#%% [markdown]
# From the box plot, the gap between Richer and Richest is not obvious. While the gap between the Middle, Poorest and Poorer is very obvious.

#%%
def plot_box(df, col, col_y = 'Case Identification'):
    sb.set_style("whitegrid")
    sb.boxplot(col, col_y, data=df)
    plt.xlabel(col) # Set text for the x axis
    plt.ylabel(col_y)# Set text for y axis
    plt.show()
    
plot_box(df, 'Region')    

#%% [markdown]
# As expected regions are distinct from each other.

#%%
def plot_box(df, col, col_y = 'Case Identification'):
    sb.set_style("whitegrid")
    sb.boxplot(col, col_y, data=df)
    plt.xlabel(col) # Set text for the x axis
    plt.ylabel(col_y)# Set text for y axis
    plt.show()
    
plot_box(df, 'Has Electricity')    

#%% [markdown]
# There is obvious difference in the number of people having and not having electricity.

#%%
def plot_box(df, col, col_y = 'Case Identification'):
    sb.set_style("whitegrid")
    sb.boxplot(col, col_y, data=df)
    plt.xlabel(col) # Set text for the x axis
    plt.ylabel(col_y)# Set text for y axis
    plt.show()
    
plot_box(df, 'Type of Place of Residence')    

#%% [markdown]
# As expected type of places of residence is also obviously distinct. 
#%% [markdown]