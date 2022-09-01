import pandas as pd
from matplotlib import pyplot as plt
from sklearn import linear_model
from sklearn.metrics import r2_score
import numpy as np


def plotear_con_var_obj(var_explicativas, var_obj):
    for var_expl_name, var_expl_values in var_explicativas.items():
        plt.scatter(var_expl_values, var_obj)
        plt.ylabel(var_obj.columns[0])
        plt.xlabel(var_expl_name)
        plt.show()


def plotear_regresiones_simples(var_explicativas, var_obj):
    for i in range(len(var_explicativas)):
        pass



def regresion_simple(var_expl, var_obj) -> linear_model.LinearRegression:
    regr = linear_model.LinearRegression()
    regr.fit(var_expl, var_obj)
    return regr


def regresion_multiple(var_explicativas, var_obj):
    pass


if __name__ == '__main__':
    startups = pd.read_csv('50_Startups.csv')
    vars_explicativas = startups.drop(columns=['Profit'])
    var_objetiva = startups[['Profit']]

    plotear_con_var_obj(vars_explicativas, var_objetiva)
