import pandas as pd
import numpy as np
import seaborn as sns
import os.path as path
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from pandas_profiling import ProfileReport
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.linear_model import LinearRegression

# ================ Understanding dataset ===================

def load_dataset():
    print('\t *** Load dataset ***\n')
    # load dataset
    boston = load_boston()

    # creating a dataframe with boston dataset
    data = pd.DataFrame(boston.data, columns=boston.feature_names)

    # add the target to predict
    data['MEDV'] = boston.target

    print('> Successfully loaded\n')
    return data

def building_pandas_profiling(data):
    
    # creating pandas profiling to understand data behavior
    profile = ProfileReport(data, title='Report - Pandas Profiling', html={'style':{'full_width':True} })
    profile.to_file('report/report_profiling.html')

# ================ Exploratory analysis ================

def building_baseline(data):
    
    print('\t *** Building a baseline ***\n')
    # converting data
    data.RM = data.RM.astype(int)

    # define rules to categorize data
    categories = []

    # fill categories
    for avarage_number_rooms in data.RM.iteritems():
        value = avarage_number_rooms[1]

        if (value <= 4):
            categories.append('Small')
        elif (value < 7):
            categories.append('Medium')
        else:
            categories.append('Large')
    
    data['categories'] = categories

    print(f'> Number of categories:\n{data.categories.value_counts()}\n')

    # agroup categories and calcule houses averages
    price_averages_by_categories = data.groupby(by='categories')['MEDV'].mean()

    print(f'> Average Price:\n {price_averages_by_categories}\n')

    dict_baseline = {
        'Large': price_averages_by_categories[0], 
        'Medium': price_averages_by_categories[1],
        'Small': price_averages_by_categories[2]
    }

    return dict_baseline

def consult_average_price(dict_baseline, n_rooms):

    if (n_rooms <= 4):
        return dict_baseline['Small']
    elif (n_rooms < 7):
        return dict_baseline['Medium']
    else:
        return dict_baseline['Large']

def creating_train_and_test_variables(data):

    # remove collinear columns, target variable and categories columns
    X = data.drop(['RAD','TAX','MEDV','DIS','AGE','ZN','categories'], axis=1)

    # set target column
    y = data['MEDV']

    # divides the data between the training and test set, 80% and 20% respectively.
    # random state can be anyone
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

    # number of rows and columns from the sets
    print (f'> Setting Dataset train and test:\n')
    print ('X_train: numbers of rows and columns: {}'.format(X_train.shape))
    print ('X_test: numbers of rows and columns: {}'.format(X_test.shape))
    print ('y_train: numbers of rows and columns: {}'.format(y_train.shape))
    print ('y_test: numbers of rows and columns: {}\n'.format(y_test.shape))

    return X_train, X_test, y_train, y_test

def baseline_predictions(data, baseline):

    print('\t *** Baseline prediction ***\n')

    X_train, X_test, y_train, y_test = creating_train_and_test_variables(data)

    # building a predict list to compare with baseline
    predict = []

    for avarage_number_rooms in X_test.RM.iteritems():
        n_room = avarage_number_rooms[1]
        predict.append(consult_average_price(baseline, n_room))

    calculate_mean_square_error(y_test, predict)

    compare_predictions(y_test, predict, 'baseline')

def compare_predictions(y_test, predict, metric_name):

    print('\t *** Compare predictions ***\n')

    df_results = pd.DataFrame()

    df_results['real_value'] = y_test.values
    df_results[f'predict_value_by_{metric_name}'] = predict

    output = 'logs/predict_logs.tsv'

    if not path.exists(output):
        df_results.to_csv(output, sep='\t', index=False)
    # just concat if exists
    else:
        log = pd.read_csv(output, sep='\t')
        log[f'predict_value_by_{metric_name}'] = predict
        log.to_csv(output, sep='\t', index=False)

    print(f'> Saving predict logs in {output}\n')

def calculate_mean_square_error(y_test, predict):
    print('\n\t *** Calculate mean square error ***\n')

    rmse = (np.sqrt(mean_squared_error(y_test, predict)))

    print(f'> Mean square error: {rmse}\n')


def predict_values_by_regression_linear(data):

    print('\t *** Linear Regression prediction ***\n')

    X_train, X_test, y_train, y_test = creating_train_and_test_variables(data)
    
    # create Linear Regression object
    lin_model = LinearRegression()

    # fit the model
    lin_model.fit(X_train, y_train)

    # evaluate model with test data 
    y_pred = lin_model.predict(X_test)

    calculate_mean_square_error(y_test, y_pred)
    compare_predictions(y_test, y_pred, 'linear_regression')

def creating_dynamic_graph():

    print('\n\t *** Creating html dynamic graph ***\n')

    input_file = 'logs/predict_logs.tsv'
    predict_log = pd.read_csv(input_file, sep='\t')

    fig = go.Figure()

    # row with test data
    fig.add_trace(go.Scatter(x=predict_log.index,
                             y=predict_log.real_value,
                             mode='lines+markers',
                             name='Real Value'))

    # row with predict data by baseline
    fig.add_trace(go.Scatter(x=predict_log.index,
                             y=predict_log.predict_value_by_baseline,
                             mode='lines+markers',
                             name=f'Predict Value Baseline'))

     # row with predict data by linear regression
    fig.add_trace(go.Scatter(x=predict_log.index,
                             y=predict_log.predict_value_by_linear_regression,
                             mode='lines+markers',
                             name=f'Predict Value Linear Regression'))

    fig.write_html('../graphs/predictions_log.html')

    print(f'> Html dynamic file successfully created\n')