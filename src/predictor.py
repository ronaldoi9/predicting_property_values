import pandas as pd
import numpy as np
import seaborn as sns
import os.path as path
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.express as px
from math import sqrt
from scipy import stats
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from pandas_profiling import ProfileReport

# ================ Understanding dataset ===================

def load_boston_dataset():
    print('=============================================\n')
    print('\t\tLoad dataset\n')
    print('=============================================\n')
    # load dataset
    boston = load_boston()

    # creating a dataframe with boston dataset
    data = pd.DataFrame(boston.data, columns=boston.feature_names)

    # add the target to predict
    data['MEDV'] = boston.target

    print('> Successfully loaded\n')
    print('=============================================\n')
    return data

def building_pandas_profiling(data):
    
    # creating pandas profiling to understand data behavior
    profile = ProfileReport(data, title='Report - Pandas Profiling', html={'style':{'full_width':True} })
    profile.to_file('data_report/report_profiling.html')

# ================ Exploratory analysis ================

def outliers_treatment(data):

    print('\t\tOutliers Treatment\n')
    print('=============================================\n')
    # statistical describe variable MEDV 
    print(f'> Describe MEDV:\n\n {data.MEDV.describe()}\n')

    # show medv distribution
    labels = ['Distribuição da variável MEDV (preço médio do imóvel)']
    fig = ff.create_distplot([data.MEDV], labels, bin_size=.2)
    fig.write_html('graphs/outliers_treatment/medv_distribution.html')

    # evaluate pearson coef.
    print(f'> Pearson coefficient: {stats.skew(data.MEDV)}\n')

    # histogram plot MEDV (target variable)
    fig = px.histogram(data, x='MEDV', nbins=50, opacity=0.50)
    fig.write_html('graphs/outliers_treatment/medv_histogram.html')

    # show MEDV outliers
    fig = px.box(data, y='MEDV')
    fig.update_layout( width=800,height=800)
    fig.write_html('graphs/outliers_treatment/medv_box_outliers.html')

    # print top 16 highest values from MEDV by RM
    print(f"> TOP 16 MEDV by RM:\n\n{data[['RM', 'MEDV']].nlargest(16, 'MEDV')}\n")
    
    # filter top 16 highest MEDV values
    top16 = data.nlargest(16, 'MEDV').index

    # remove all top16 values (outliers values)
    data.drop(top16, inplace=True)
    print(f'> Removing outliers...\n')

    # show medv distribution after remove outliers values
    labels = ['Distribuição da variável MEDV (depois da remoção de outliers)']
    fig = ff.create_distplot([data.MEDV], labels, bin_size=.2)
    fig.write_html('graphs/outliers_treatment/medv_distribution_after_remove_outliers.html')

    # # histogram plot MEDV after remove outliers values (target variable)
    fig = px.histogram(data, x="MEDV", nbins=50, opacity=0.50)
    fig.write_html('graphs/outliers_treatment/medv_histogram_after_remove_outliers.html')

    print(f'> Pearson coefficient after treatment: {stats.skew(data.MEDV)}\n')
    print('=============================================\n')

# ================ Deploy ================
def building_baseline(data):
    
    print('\t\tBuilding a baseline\n')
    print('=============================================\n')
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

    print(f'> Number of categories:\n\n{data.categories.value_counts()}\n')

    # agroup categories and calcule houses averages
    price_averages_by_categories = data.groupby(by='categories')['MEDV'].mean()

    print(f'> Baseline Average Prices:\n\n{price_averages_by_categories}\n')

    dict_baseline = {
        'Large': price_averages_by_categories[0], 
        'Medium': price_averages_by_categories[1],
        'Small': price_averages_by_categories[2]
    }

    print('=============================================\n')
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

    print('\t\tBaseline prediction\n')
    print('=============================================\n')

    X_train, X_test, y_train, y_test = creating_train_and_test_variables(data)

    # building a predict list to compare with baseline
    predict = []

    for avarage_number_rooms in X_test.RM.iteritems():
        n_room = avarage_number_rooms[1]
        predict.append(consult_average_price(baseline, n_room))

    compare_predictions(y_test, predict, 'baseline')

    return calculate_mean_square_error(y_test, predict)

def compare_predictions(y_test, predict, metric_name):

    print('\t\t*** Compare predictions ***\n')

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
    print('\n\t*** Calculate mean square error ***\n')

    rmse = (np.sqrt(mean_squared_error(y_test, predict)))

    print(f'> Mean square error: {rmse}\n')
    print('=============================================\n')
    return rmse

def predict_values_by_linear_regression(data):

    print('\t\tLinear Regression prediction\n')
    print('=============================================\n')

    X_train, X_test, y_train, y_test = creating_train_and_test_variables(data)
    
    # create Linear Regression object
    lin_model = LinearRegression()

    # fit the model
    lin_model.fit(X_train, y_train)

    # evaluate model with test data 
    y_pred = lin_model.predict(X_test)

    compare_predictions(y_test, y_pred, 'linear_regression')

    return calculate_mean_square_error(y_test, y_pred)

def predict_values_by_decision_tree(data):

    print('\t\tDecision Tree prediction\n')
    print('=============================================\n')

    X_train, X_test, y_train, y_test = creating_train_and_test_variables(data)

    decision_tree = DecisionTreeRegressor()

    # fit model
    decision_tree.fit(X_train, y_train)

    # evaluate model with test data
    y_pred = decision_tree.predict(X_test)

    compare_predictions(y_test, y_pred, 'decision_tree')

    return calculate_mean_square_error(y_test, y_pred)

def predict_value_by_random_forest(data):

    print('\t\tDecision Tree prediction\n')
    print('=============================================\n')

    X_train, X_test, y_train, y_test = creating_train_and_test_variables(data)

    random_forest = RandomForestRegressor()

    # fit model
    random_forest.fit(X_train, y_train)

    # evaluate model with test data
    y_pred = random_forest.predict(X_test)

    compare_predictions(y_test, y_pred, 'random_forest')

    return calculate_mean_square_error(y_test, y_pred)

def creating_dynamic_graph():

    print('\t\tCreating html dynamic graph\n')
    print('=============================================\n')

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

    # row with predict data by decision tree
    fig.add_trace(go.Scatter(x=predict_log.index,
                             y=predict_log.predict_value_by_decision_tree,
                             mode='lines+markers',
                             name='Predict Value Decision Tree'))

    # row with predict data by random forest
    fig.add_trace(go.Scatter(x=predict_log.index,
                             y=predict_log.predict_value_by_random_forest,
                             mode='lines+markers',
                             name='Predict Value Random Forest'))

    fig.write_html('graphs/predictions/predictions_log.html')

    print(f'> Html dynamic file successfully created\n')
    print('=============================================\n')