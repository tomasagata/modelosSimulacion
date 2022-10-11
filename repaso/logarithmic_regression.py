import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

class LogaritmicRegression:
    def __init__(self, X, y):
        if 'const' in X.columns:
            Xl = X.drop('const', axis=1)
        else:
            Xl = X
        Xl = np.log10(Xl)
        Xl = sm.add_constant(Xl)
        yl = np.log10(y)

        self.model = sm.OLS(yl, Xl).fit()
        self.linear_transform_params = self.model.params
        self.params = [
            np.power(10, self.linear_transform_params[0]),
            self.linear_transform_params[1]
        ]
    
    def predict(self, X):
        return self.params[0] * np.power(X, self.params[1])

df = pd.read_csv(r'C:\Users\tomas\Desktop\modelosSimulacion\data\boston.csv')
set_vars_expl = df[['LSTAT']]
var_obj = df['MEDV']

model = LogaritmicRegression(set_vars_expl, var_obj)

plt.scatter(set_vars_expl['LSTAT'], var_obj)
plt.scatter(set_vars_expl['LSTAT'], model.predict(set_vars_expl), c='r')
plt.show()
