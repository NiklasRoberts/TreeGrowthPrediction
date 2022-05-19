# AI-Final-Project
Using applied data science techniques to analyze a Forest Ecology dataset measuring bark infections over several years.


Years:
1982
1985
1990

Table columns:
- Tree num
- D82
- D85
- D90
- Nutrients at 1982
- Nutrients at 1985


Tree growth per year:

From 1982 -> 1985
- tree num
- (year)
- (D increase/dprev) / year * 100
- nutrients at 1982

From 1985 -> 1990
- tree num
- (year)
-(D increase/dprev) / year * 100
- nutrients at 1985



Assumptions:
- Trees growth rate is not affected by the trees age
- Foliar nutrient levels are measured at the beginning of the growing time period
- Foliar nutrients remain constant over growth period
- Foliar nutrients for a plot is the same for all trees within that plot
- The same tree may have growth/yr calculated from it twice (once from 82-85, once from 85-90)



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




Regression Model:






Some sources so far:

Possible ways to increase accuracy of neural network
https://towardsdatascience.com/how-to-increase-the-accuracy-of-a-neural-network-9f5d1c6f407d

How to choose activation function
https://machinelearningmastery.com/choose-an-activation-function-for-deep-learning/

Batch size and epochs: 
https://towardsdatascience.com/epoch-vs-iterations-vs-batch-size-4dfb9c7ce9c9#:~:text=Note%3A%20The%20number%20of%20batches,iterations%20to%20complete%201%20epoch.

Good source on layers and nodes:
https://machinelearningmastery.com/how-to-configure-the-number-of-layers-and-nodes-in-a-neural-network/