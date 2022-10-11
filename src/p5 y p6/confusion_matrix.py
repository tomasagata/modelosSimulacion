import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
import numpy as np


def confusion_matrix(model, X, y, threshold = 0.5):
    y_proba = model.predict_proba(X)
    y_size = len(y_proba)
    tp = 0; fp = 0; fn = 0; tn = 0
    for i in range(y_size):
        y_pred = 0
        if y_proba[i][1] > threshold:
            y_pred = 1
        
        if y_pred == 1 and y[i] == 1:
            tp += 1
        elif y_pred == 1 and y[i] == 0:
            fp += 1
        elif y_pred == 0 and y[i] == 1:
            fn += 1
        elif y_pred == 0 and y[i] == 0:
            tn += 1
    return tp, fp, fn, tn

def expected_benefits(model, X, y, threshold = 0.5, tp_benefits = 1, fp_benefits = -1, fn_benefits = -1, tn_benefits = 1):
    tp, fp, fn, tn = confusion_matrix(model, X, y, threshold)
    return tp_benefits * tp + fp_benefits * fp + fn_benefits * fn + tn_benefits * tn


def factorize(df, var_name_arr):
    new_df = df
    for var_name in var_name_arr:
        var = pd.factorize(new_df[var_name])
        var_index = new_df.columns.get_loc(var_name)
        new_df = new_df.drop(var_name, axis=1)
        new_df.insert(var_index, var_name, var[0])
    return new_df

if __name__ =='__main__':
    df = pd.read_csv(r'C:\Users\tomas\Desktop\modelosSimulacion\data\Churn_Modelling.csv')
    df = df.drop(['RowNumber','CustomerId','Surname'], axis=1)
    set_vars_expl = df.drop(['Exited'], axis=1)
    set_vars_expl = factorize(set_vars_expl, ['Geography', 'Gender'])
    var_obj = df['Exited']

    model = tree.DecisionTreeClassifier(min_samples_leaf=5).fit(set_vars_expl, var_obj)
    
    for i in np.arange(0,1,0.1):
        print(expected_benefits(model, set_vars_expl, var_obj, i))