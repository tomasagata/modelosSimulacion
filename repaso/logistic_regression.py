import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

def predict(y_proba, t=0.5):
    y_pred = []
    for proba in y_proba:
        pred = 0
        if proba[1] > t:
            pred = 1

        y_pred.append(pred)
    return pd.Series(y_pred)

def mse(y_pred, y_real):
    return (np.power(y_pred - y_real, 2)).mean()

def factorize(df, var_name_arr):
    new_df = df
    for var_name in var_name_arr:
        var = pd.factorize(new_df[var_name])
        index = new_df.columns.get_loc(var_name)
        new_df = new_df.drop(var_name, axis=1)
        new_df.insert(index, var_name, var[0])
    return new_df

df = pd.read_csv(r'/Users/kpapiccio/PycharmProjects/modelosSimulacion/data/Churn_Modelling.csv')
df = df.drop(['RowNumber','CustomerId','Surname'], axis=1)
set_vars_expl = df.drop('Exited', axis=1)
set_vars_expl = factorize(set_vars_expl, ['Geography', 'Gender'])
var_obj = df['Exited']

model = LogisticRegression().fit(set_vars_expl, var_obj)
y_proba = model.predict_proba(set_vars_expl)

for t in np.arange(0,1,0.1):
    y_pred = predict(y_proba, t)
    print('threshold: ' + str(t))
    print('mse: ' + str(mse(y_pred, var_obj)))
    print('accuracy: ' + str(1 - mse(y_pred, var_obj)))
    print('------------------------------------------')
