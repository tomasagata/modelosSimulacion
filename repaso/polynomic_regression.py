import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

class PolynomicRegression:
    def __init__(self, X: pd.DataFrame, y, n: int):
        if 'const' in X.columns:
            Xp = X.drop('const', axis=1)
        else:
            Xp = X
        Xp = PolynomialFeatures(n).fit_transform(Xp)
        Xp = sm.add_constant(Xp)

        print(Xp)
        self.n = n
        self.model = sm.OLS(y, Xp).fit()
        self.params = self.model.params
        self.rsquared_adj = self.model.rsquared_adj
    
    def predict(self, X: pd.DataFrame):
        if 'const' in X.columns:
            Xp = X.drop('const', axis=1)
        else:
            Xp = X

        Xp = PolynomialFeatures(self.n).fit_transform(Xp)
        return self.model.predict(Xp)
    
    def summary(self):
        return self.model.summary()

def mse(y_real, y_pred):
    return (np.power(y_real - y_pred, 2)).mean()

df = pd.read_csv(r'C:\Users\tomas\Desktop\modelosSimulacion\data\boston.csv')
set_vars_expl = df[['LSTAT']]
var_obj = df['MEDV']

model = PolynomicRegression(set_vars_expl, var_obj, 3)
plt.scatter(set_vars_expl['LSTAT'], var_obj)
plt.scatter(set_vars_expl['LSTAT'], model.predict(set_vars_expl), c='r')
plt.show()
