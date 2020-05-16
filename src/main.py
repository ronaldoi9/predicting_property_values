import predictor

if __name__ == '__main__':

    data = predictor.load_dataset()
    # predictor.building_pandas_profiling(data)
    baseline = predictor.building_baseline(data)
    predictor.baseline_predictions(data, baseline)
    predictor.predict_values_by_regression_linear(data)

    # predictor.creating_dynamic_graph()