# Escribir un programa que me prediga mediante aproximaciones lineales el valor medio de
# una propiedad segun el 'LSTAT' y 'RM' de la base de datos boston. Imprimir el mse y graficar
# en 3 dimensiones la aproximacion obtenida

from matplotlib import projections
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

def mse(y_pred, y_real):
    return (np.power(y_pred - y_real, 2)).mean()

df = pd.read_csv(r'C:\Users\tomas\Desktop\modelosSimulacion\data\boston.csv')
set_vars_expl = df[['LSTAT', 'RM']]
var_obj = df['MEDV']

model = LinearRegression().fit(set_vars_expl, var_obj)

print('Coeficientes: ' + str(model.coef_) + '\nIntersecciones: ' + str(model.intercept_))
print('mse: ' + str(mse(model.predict(set_vars_expl), var_obj)))

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(set_vars_expl['RM'], set_vars_expl['LSTAT'], var_obj)
ax.plot_trisurf(set_vars_expl['RM'], set_vars_expl['LSTAT'], model.predict(set_vars_expl), color='r')
plt.show()