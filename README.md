# Predicting property values :chart_with_upwards_trend:

## About this Project
The main idea is:

*"Apply Machine Learn to predict Boston houses prices"*

In this project used machine learning to predict Boston property prices, and to show the data was build a data app, able to interact and receive user information and predict the property prices from the input data.

Data app can be find [here](https://predicting-property-values.herokuapp.com/). (*All informations are portuguese language* <span>&#x1f1e7;&#x1f1f7;</span>)

## About dataset
This dataset is about Boston houses prices, and can be get through sklearn library, you can get more information [here](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_boston.html)

**Attribute Information:**

- **CRIM**:     per capita crime rate by town
- **ZN**:       proportion of residential land zoned for lots over 25,000 sq.ft.
- **INDUS**:    proportion of non-retail business acres per town
- **CHAS**:     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
- **NOX**:      nitric oxides concentration (parts per 10 million)
- **RM**:       average number of rooms per dwelling
- **AGE**:      proportion of owner-occupied units built prior to 1940
- **DIS**:      weighted distances to five Boston employment centres
- **RAD**:      index of accessibility to radial highways
- **TAX**:      full-value property-tax rate per $10,000
- **PTRATIO**:  pupil-teacher ratio by town
- **B**:        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
- **LSTAT**:    % lower status of the population
- **MEDV**:     Median value of owner-occupied homes in $1000's

## Solution
To predict a estimated value, i start a simple baseline, and apply Machine Learning algorithms like Linear Regression, Decision Tree and Random Forest to get better results.

The baseline metric is very simple (*shown in code*). Therefore so his prediction isn't so good, below the figure show a comparison between the Real Value and Baseline Prediction

![baseline](https://user-images.githubusercontent.com/40616142/82163103-628e8f80-987f-11ea-82bd-a42f7536399a.jpg)

For bests results, i apply Machine Learn algorithms, in order to get close of Real Value function.

- ### **Linear Regression**
![linear_regression](https://user-images.githubusercontent.com/40616142/82163388-3247f080-9881-11ea-9181-fca322c78bda.jpg)

- ### **Decision Tree**
![decision_tree](https://user-images.githubusercontent.com/40616142/82163468-ca45da00-9881-11ea-8c2b-e7fb4ea5b57a.jpg)

- ### **Random Forest**
![random_forest](https://user-images.githubusercontent.com/40616142/82163483-e0ec3100-9881-11ea-999a-b15134a4b51d.jpg)

### **Errors**
To measure the real performance of the algorithms, the mean squared error is calculated.

- Baseline error: 6.21
- Linear Regression error: 4.46
- Dicision Tree error: 4.42
- Random Forest error: 3.33

The **Random Forest** algorithms was the best results along others. Then he will be the algorithm used to build the data app.

## Getting started


## Conclusion

