import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def mse(y_pred, y_real):
    return (np.power(y_pred - y_real, 2)).mean()

df = pd.read_csv(r'C:\Users\tomas\Desktop\modelosSimulacion\data\boston.csv')
var_exp = df[['LSTAT']]
var_obj = df['MEDV']

model = LinearRegression().fit(var_exp, var_obj)

print('beta 0: ' + str(model.coef_) + '\nbeta 1: ' + str(model.intercept_))
print('mse: ' + str(mse(model.predict(var_exp), var_obj)))

plt.scatter(var_exp['LSTAT'], var_obj)
plt.plot(var_exp['LSTAT'], model.predict(var_exp), c='r')
plt.show()