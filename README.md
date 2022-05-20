# AI-Final-Project
Using applied data science techniques to analyze a [Forest Ecology dataset](https://data.nal.usda.gov/dataset/c-55-thinning-and-fertilization-western-redcedar-c55-wrc-tf) measuring tree growth in response to thinning and fertilization in the western Olympic Peninsula over several years.

## Environment setup
- Create fresh environment using python=3.8
- `pip install -r requirements.txt`

# Methodology

## Analysis target
Our goal was to calculate and predict the average growth per year of all trees measured in the study. We derived `Growth%/Yr` using tree width measurements from 1982, 1985, and 1990 and measuring the percent change across those time periods. We then used a variety of neural network approaches to predict the `Growth%/Yr` based on a variety of input variables.


## Input variables
Several measures were used to predict tree growth:
- Tree species
- Nutrient levels in fallen foliage:
  - Nitrogen concentration (%)
  - Phosphorus concentration (%)
  - Potassium concentration (%)
  - Sulfur concentration (%)
  - Calcium concentration (%)

## Assumptions:
- Trees growth rate is not affected by the trees age
- Foliar nutrient levels are measured at the beginning of the growing time period
- Foliar nutrients remain constant over growth period
- Foliar nutrients for a plot is the same for all trees within that plot
- The same tree may have `Growth%/Yr` calculated from it twice (once from 82-85, once from 85-90)

## Data derivation

### **Percent growth per year**:
- D82 = 1982 Tree diameter at breast height
- D85 = 1985 Tree diameter at breast height
- D90 = 1990 Tree diameter at breast height
 
1982-1985 period `Growth%/Yr` = (D85 - D82)/(D82 * 3 Yrs)

1985-1990 period `Growth%/Yr` = (D90 - D85)/(D85 * 5 Yrs)
  
### **Foliar nutrients**:
- 1982 measurements associated with tree growth during 1982-1985 
- 1985 measurements associated with tree growth during 1985-1990

### **Tree species**:
- THPL
- TSHE
- PISI
- ALRU2
- RHPU
- TABR2
- UNCLH 
- ABAM
- ABGR
- MAFU
- TSME

# Statistical Analysis

## Models

### Regression


### Binning






Different Models:

- Growth per year bins (2 approaches):
    - Bin by fixed range (0-.2, .2-.4, .4-.6, .6+)
    - Bin by equal record distribution

- Regression
    - one output node, coninuous range


Classification Model:
    
Initial model:
    classifyModel = Sequential()    
    classifyModel.add(Dense(4, input_dim=6, activation='sigmoid'))
    classifyModel.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    - training accuracy: 0.4927
    - test accuracy:     0.4950

Increasing accuracy test:
- messed around with batch size and epochs -> didn't result in to much accuracy difference
- changed activation to relu:
    - training accuracy: 0.2696
    - test accuracy:     0.2520
- changed activation to tanh
    - training accuracy: 0.3352
    - test accuracy:     0.3432
- changed activation to softmax
    - training accuracy: 0.5029
    - test accuracy:     0.4905
- read that relus are good for hidden layers, so added a hidden layer, with sigmoid as final output activation as below:
    classifyModel = Sequential()
    classifyModel.add(Dense(8, input_dim=6, activation='relu'))
    classifyModel.add(Dense(4, activation='sigmoid'))
    classifyModel.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    - training accuracy: 0.5105
    - test accuracy:     0.4869


## Helpful sources:

Possible ways to increase accuracy of neural network
https://towardsdatascience.com/how-to-increase-the-accuracy-of-a-neural-network-9f5d1c6f407d

How to choose activation function
https://machinelearningmastery.com/choose-an-activation-function-for-deep-learning/

Batch size and epochs: 
https://towardsdatascience.com/epoch-vs-iterations-vs-batch-size-4dfb9c7ce9c9#:~:text=Note%3A%20The%20number%20of%20batches,iterations%20to%20complete%201%20epoch.

Good source on layers and nodes:
https://machinelearningmastery.com/how-to-configure-the-number-of-layers-and-nodes-in-a-neural-network/