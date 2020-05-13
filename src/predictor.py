import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from pandas_profiling import ProfileReport

# load dataset
boston = load_boston()

# creating a dataframe with boston dataset
data = pd.DataFrame(boston.data, columns=boston.feature_names)

# add the target to predict
data['MEDV'] = boston.target

# analysing datas
profile = ProfileReport(data, title='Report - Pandas Profiling', html={'style':{'full_width':True} })

# saving report html
profile.to_file('report/report_profiling.html')


