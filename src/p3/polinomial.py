import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import numpy as np


class PolynomialModel:
    def __init__(self, X, y, n):
        self.n = n
        if 'const' in X.columns:
            x_pol = X.drop(['const'], axis=1)
        else:
            x_pol = X
        
        x_pol = PolynomialFeatures(degree=n).fit_transform(x_pol)
        x_pol = sm.add_constant(x_pol)
        
        self.linear_model = sm.OLS(y, x_pol).fit()
        self.params = self.linear_model.params
    
    def predict(self, X):
        new_x = sm.add_constant(PolynomialFeatures(degree=self.n).fit_transform(X.values))
        return self.linear_model.predict(new_x)

if __name__ == '__main__':
    boston = pd.read_csv(r"/Users/kpapiccio/PycharmProjects/modelosSimulacion/data/boston.csv")
    set_vars_expl = boston[['LSTAT']]
    var_obj = boston[['MEDV']]

    model = PolynomialModel(set_vars_expl, var_obj, 3)

    plt.scatter(set_vars_expl.values, var_obj)
    plt.scatter(set_vars_expl.values, model.predict(set_vars_expl), c='r')
    plt.show()
