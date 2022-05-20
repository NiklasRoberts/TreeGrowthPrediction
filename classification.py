print("Importing relevant packages.")
import pandas as pd
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

# Get tree measurements for year 82, 85, and 90
print("Importing and cleaning tree measurements.")
treeMeasurements = pd.read_csv("C55_TreeMeasurements.csv")

#getting trees from 1982 and 1985 measurements
desiredCols1 = ["PLOT", "TREE", "RECRUIT", "SPECIES", "D82", "D85"]
customMeasurements1 = treeMeasurements[desiredCols1]

#getting trees from 1985 to 1990
desiredCols2 = ["PLOT", "TREE", "RECRUIT", "SPECIES", "D85", "D90"]
customMeasurements2 = treeMeasurements[desiredCols2]

#cleaning data to gather trees that only have measurements at both ends of the period
clean82_85 = customMeasurements1.dropna()
clean85_90 = customMeasurements2.dropna()

# getting foliar nutrients from csv file
foliarNutrients = pd.read_csv("C55_FoliarNutrients.csv")
foliarNutrients = foliarNutrients.drop("REPLICATE", axis=1)

# Get foliar nutrient levels 1982
foliarNutrients82 = foliarNutrients.loc[foliarNutrients['YEAR'] == 1982]

# Get foliar nutrient levels 1985
foliarNutrients85 = foliarNutrients.loc[foliarNutrients['YEAR'] == 1985]

# Add foliar nutrients to tree data 
allTreeData82_85 = pd.merge(clean82_85, foliarNutrients82, on="PLOT", how="outer")
allTreeData85_90 = pd.merge(clean85_90, foliarNutrients85, on="PLOT", how="outer")

# Getting percent growth for each tree per year
# perc_growth = (D85-D82)/D82
allTreeData82_85["Growth"] = (allTreeData82_85["D85"]-allTreeData82_85["D82"])/allTreeData82_85["D82"]
# growth_per_yr = prec_growth/85-82
allTreeData82_85["Growth/yr"] = allTreeData82_85["Growth"]/3

#getting percent growth for 1985-1990
# perc_growth = (D90-D85)/D825
allTreeData85_90["Growth"] = (allTreeData85_90["D90"]-allTreeData85_90["D85"])/allTreeData85_90["D85"]
# growth_per_yr = prec_growth/85-82
allTreeData85_90["Growth/yr"] = allTreeData85_90["Growth"]/5

# Extract relevant columns: 
relColumns = ["SPECIES", "TOTN", "TOTP", "TOTK", "TOTS", "TOTCa", "Growth/yr"]
modelingData82 = allTreeData82_85[relColumns]
modelingData85 = allTreeData85_90[relColumns]

# combine dataframes to make one dataframe with all data needed
modelingData = pd.concat([modelingData82, modelingData85], ignore_index=True)
modelingData['id'] = modelingData.index

# Getting Columns for neural network classification
# What bin does the growth/yr fall into? that bin gets a 1, others get 0

mergeDF = pd.DataFrame(columns=['id', 'bin'])

# For each row in modelingData setting a bin based off how growth/year
for index, row in modelingData.iterrows():
    #bins: 0-0.05, 0.05+ to 0.1, 0.1+ to 0.2, 0.2+
    if row['Growth/yr'] <= .05:
        mergeDF.loc[index] = pd.Series({'id': index, 'bin': 0})
    elif row['Growth/yr'] <= .1:
        mergeDF.loc[index] = pd.Series({'id': index, 'bin': 1})
    elif row['Growth/yr'] <= .2:
        mergeDF.loc[index] = pd.Series({'id': index, 'bin': 2})
    else:
        mergeDF.loc[index] = pd.Series({'id': index, 'bin': 3})

# joining data to include bin
modelingData = pd.merge(modelingData, mergeDF, on='id', how="outer")

# Getting data ready for the Classification Neural Network

# Input variables: SPECIES, TOTN, TOTP, TOTK, TOTS, TOTCa, and TOTMg (not used)
# label: fixed bins (1 for correct bin, 0 for all others)
classifyColumns = ['SPECIES', 'TOTN', 'TOTP', 'TOTK', 'TOTS', 'TOTCa', 'bin']
classifyData = modelingData[classifyColumns]

# for turning 'species' column into numerical data
speciesLabEncoder = LabelEncoder()

# Transforms species into numerical data
speciesNumList = speciesLabEncoder.fit_transform(classifyData['SPECIES'])
# code for reverting: species = le.inverse_transform(label)
# changing the species column to numerical values
classifyData['SPECIES'] = speciesNumList

# Splitting into input and output data
classifyX = classifyData.iloc[:,:6].values
classifyY = classifyData.iloc[:,6:7].values

# Transforming output into binary values
ohe = OneHotEncoder()
classifyY = ohe.fit_transform(classifyY).toarray()

# Splitting data into training and test data
classifyX_train, classifyX_test, classifyY_train, classifyY_test = train_test_split(classifyX, classifyY, test_size = 0.2)

#Building and Compiling Neural Network for Classification


classifyModel = Sequential()
classifyModel.add(Dense(12, input_dim=6, activation='relu'))
classifyModel.add(Dense(12, activation='relu'))
classifyModel.add(Dense(4, activation='sigmoid'))

loss_function = keras.losses.MeanAbsoluteError()
classifyModel.compile(loss=loss_function, optimizer='adam', metrics=['accuracy'])

#Training classification model
classifyModel.fit(classifyX_train, classifyY_train, epochs=100, batch_size=64)

#Testing classification model

prediction = classifyModel.predict(classifyX_test)
print(prediction)


# undoing onehot encoder
# Each line represents the score given to each class, so we only want to keep the max (best) class
pred = list()
for i in range(len(prediction)):
    pred.append(np.argmax(prediction[i]))
test = list()
for i in range(len(classifyY_test)):
    test.append(np.argmax(classifyY_test[i]))

# Determining accuracy of Classification Neural Network model
classifyAccuracy = accuracy_score(pred,test)
print('Accuracy is:', classifyAccuracy)