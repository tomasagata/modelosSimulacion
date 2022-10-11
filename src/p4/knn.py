import pandas as pd
from sklearn import neighbors
import matplotlib.pyplot as plt
import numpy as np


def factorize(df, var_name_arr):
    new_df = df
    for var_name in var_name_arr:
        var = pd.factorize(new_df[var_name])
        index = new_df.columns.get_loc(var_name)
        new_df = new_df.drop(var_name, axis=1)
        new_df.insert(index, var_name, var[0])
    
    return new_df

def colorPlot(x, y, colorVariable, x_name='X', y_name='Y'):
    plt.scatter(x,y,c=colorVariable)
    plt.colorbar()
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.show()

def knnScore(model, X, y, threshold = 0.5):
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

if __name__ == '__main__':
    df = pd.read_csv(r'/Users/kpapiccio/PycharmProjects/modelosSimulacion/data/Churn_Modelling.csv')
    df = df.drop(['RowNumber','CustomerId','Surname'], axis=1)
    set_vars_expl = df.drop('Exited', axis=1)
    set_vars_expl = factorize(set_vars_expl, ['Gender', 'Geography'])
    var_obj = df['Exited']

    model = neighbors.KNeighborsClassifier().fit(set_vars_expl, var_obj)
    
    for i in np.arange(0, 1, 0.1):
        print(knnScore(model, set_vars_expl, var_obj, i))