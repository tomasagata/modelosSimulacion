from multiprocessing import reduction
from sklearn import tree
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def factorize(dataFrame, column_name_arr):
    new_df = dataFrame
    for column_name in column_name_arr:
        var = pd.factorize(new_df[column_name])
        var_index = new_df.columns.get_loc(column_name)
        new_df = new_df.drop(column_name, axis=1)
        new_df.insert(var_index, column_name, var[0])
    
    return new_df

def decisionTreeScore(model, X, y, threshold = 0.5):
    y_proba = model.predict_proba(X)
    y_size = len(y_proba)
    hits = 0
    for i in range(y_size):
        y_pred = 0
        if y_proba[i][1] > threshold:
            y_pred = 1
        
        if y_pred == y[i]:
            hits += 1
    return hits/y_size

def plotTree(model):
    plt.figure(figsize=(20,20))
    tree.plot_tree(model, fontsize=10)
    plt.show()

if __name__ == '__main__':
    df = pd.read_csv(r'C:\Users\tomas\Desktop\modelosSimulacion\data\Churn_Modelling.csv')
    df = df.drop(['RowNumber','CustomerId','Surname'], axis=1)
    set_vars_expl = df.drop('Exited', axis=1)
    set_vars_expl = factorize(set_vars_expl, ['Geography', 'Gender'])
    var_obj = df['Exited']

    model = tree.DecisionTreeClassifier(min_samples_leaf=5).fit(set_vars_expl, var_obj)

    for threshold in np.arange(0, 1, 0.1):
        print(decisionTreeScore(model, set_vars_expl, var_obj, threshold))
    print(model.score(set_vars_expl, var_obj))
    # plotTree(model)

