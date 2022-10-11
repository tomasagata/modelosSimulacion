import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt


class LogarithmicRegression:

    def __init__(self, X, y):
        if 'const' in X.columns:
            x_log = np.log10(X.drop(['const'], axis=1))
        else:
            x_log = np.log10(X)
        
        x_log = sm.add_constant(x_log)
        y_log = np.log10(y)
        self.linear_model = sm.OLS(y_log, x_log).fit()
        self.params = {
            'beta_0': self.linear_model.params[0],
            'beta_1': self.linear_model.params[1],
            'a': np.power(10, self.linear_model.params[0]),
            'b': self.linear_model.params[1]
        }

    def predict(self, X):
        return self.params['a'] * np.power(X, self.params['b'])
    


if __name__ == '__main__':
    boston = pd.read_csv(r"/Users/kpapiccio/PycharmProjects/modelosSimulacion/data/boston.csv")
    set_vars_expl = boston[['LSTAT']]
    var_obj = boston[['MEDV']]

    model = LogarithmicRegression(set_vars_expl, var_obj)

    x = np.arange(start=1, stop=40, step=0.0625)

    plt.scatter(set_vars_expl['LSTAT'], var_obj)
    plt.scatter(x, model.predict(x), c='r')
    plt.show()

