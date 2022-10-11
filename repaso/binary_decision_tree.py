from matplotlib.pyplot import axes
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.metrics import confusion_matrix


def factorize(df, var_name_arr):
    new_df = df
    for var_name in var_name_arr:
        var = pd.factorize(new_df[var_name])
        index = new_df.columns.get_loc(var_name)
        new_df = new_df.drop(var_name, axis=1)
        new_df.insert(index, var_name, var[0])
    return new_df

def mse(y_pred, y_real):
    return (np.power(y_pred - y_real, 2)).mean()

def predict(y_proba, t=0.5):
    y_pred = []
    for proba in y_proba:
        pred = 0
        if proba[1] > t:
            pred = 1
        
        y_pred.append(pred)
    return pd.Series(y_pred)

df = pd.read_csv(r'/Users/kpapiccio/PycharmProjects/modelosSimulacion/data/Churn_Modelling.csv')
df = df.drop(['RowNumber','CustomerId','Surname'], axis=1)
set_var_expl = df.drop('Exited', axis=1)
set_var_expl = factorize(set_var_expl, ['Geography', 'Gender'])
var_obj = df['Exited']

model = tree.DecisionTreeClassifier(min_samples_leaf=5).fit(set_var_expl, var_obj)
y_proba = model.predict_proba(set_var_expl)

for t in np.arange(0,1,0.1):
    y_pred = predict(y_proba, t)
    mse_val = mse(y_pred, var_obj)
    print('Threshold: '+ str(t))
    print('mse: ' + str(mse_val))
    print('accuracy: ' + str(1 - mse_val))
    print(confusion_matrix(var_obj, y_pred))
    print('-------------------------------\n')