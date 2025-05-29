from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from pipeline import build_final_pipeline
from sklearn.model_selection import train_test_split
import pandas as pd
from load_data import load_data
'''
    evaluate the model predictions
'''

whole_df = load_data()
x = whole_df.drop(columns=['Price'])
y = whole_df['Price']


df_test = pd.read_csv('src/test.csv')
x_test = df_test.drop(columns=['Price'])
y_test = df_test['Price']

df_train = pd.read_csv('src/train.csv')
x_train = df_train.drop(columns=['Price'])
y_train = df_train['Price']

model_pipeline = build_final_pipeline()
y_pred = model_pipeline.predict(x_test)

MSError = mean_squared_error(y_pred, y_test) 
cross_validation_score = cross_val_score(model_pipeline, x_train, y_train, cv=5)


