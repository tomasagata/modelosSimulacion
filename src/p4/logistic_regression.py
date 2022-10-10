from tracemalloc import start
import pandas as pd
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np


def factorize(df, variable_name_list):
    new_df = df
    for var_name in variable_name_list:
        var = pd.factorize(new_df[var_name])
        index = new_df.columns.get_loc(var_name)
        new_df = new_df.drop([var_name], axis=1)
        new_df.insert(index, var_name, var[0])

    return new_df

def colorPlot(x, y, colorVariable, x_name='X', y_name='Y'):
    plt.scatter(x,y,c=colorVariable)
    plt.colorbar()
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.show()

def logisticRegressionScore(model, X, y, threshold = 0.5):
    y_proba = model.predict_proba(set_vars_expl)
    y_size = len(y_proba)
    hits = 0
    for i in range(y_size):
        if y_proba[i][1] > threshold:
            y_pred = 1
        else:
            y_pred = 0
        
        if y_pred == y[i]:
            hits += 1
    return hits/y_size


if __name__ == '__main__':
    df = pd.read_csv(r'C:\Users\tomas\Desktop\modelosSimulacion\data\Churn_Modelling.csv')
    df = df.drop(['RowNumber','CustomerId','Surname'], axis=1)
    set_vars_expl = df.drop('Exited', axis=1)
    set_vars_expl = factorize(set_vars_expl, ['Gender', 'Geography'])
    var_obj = df['Exited']

    # Pide que var_obj sea un array unidimensional, por eso el df['Exited'] de arriba
    model = LogisticRegression().fit(set_vars_expl, var_obj)

    for i in np.arange(0, 1.0625, 0.0625):
        print('threshold = ' + str(round(i, 4)) + ', score = ' + str(logisticRegressionScore(model, set_vars_expl, var_obj, i)))
    print(model.score(set_vars_expl, var_obj))
    # colorPlot(set_vars_expl['NumOfProducts'], set_vars_expl['Age'], var_obj, x_name='NumOfProducts', y_name='Age')
