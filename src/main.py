import predictor

if __name__ == '__main__':

    data = predictor.load_boston_dataset()
    #predictor.building_pandas_profiling(data)
    predictor.outliers_treatment(data)
    baseline = predictor.building_baseline(data)
    # verify what is the best prediction metric
    rmse = { }
    rmse['baseline_error'] = predictor.baseline_predictions(data, baseline)
    rmse['liner_regression_error'] = predictor.predict_values_by_linear_regression(data)
    rmse['decision_tree_error'] = predictor.predict_values_by_decision_tree(data)
    rmse['random_forest_error'] = predictor.predict_value_by_random_forest(data)

    # save dynamic graph in html file
    predictor.creating_dynamic_graph()

    print('\t\t Errors by metric\n')
    print('=============================================\n')
    print(f'> Baseline error: {rmse["baseline_error"]}\n')
    print(f'> Linear Regression error: {rmse["liner_regression_error"]}\n')
    print(f'> Dicision Tree error: {rmse["decision_tree_error"]}\n')
    print(f'> Random Forest error: {rmse["random_forest_error"]}\n')
    print('=============================================\n')

    # remove all unnecessary columns for data app
    data.drop(['RAD','TAX','DIS','AGE','ZN','categories'], axis=1, inplace=True)
    
    # saving data on disk
    data.to_csv('model/boston_data.tsv', sep='\t', index=False)