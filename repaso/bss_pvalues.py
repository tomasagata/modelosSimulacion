# A traves de la base de datos de boston, utilizar un modelo de regresion lineal que pueda predecir el MEDV
# para ello, utilizar el algoritmo bss con p-values para encontrar las variables optimas del modelo de datos. 
# Imprimir las variables utilizadas, los coeficientes y finalmente el mse del modelo

import pandas as pd
import numpy as np
import statsmodels.api as sm

def mse(y_pred, y_real):
    return (np.power(y_pred - y_real, 2)).mean()


def bss_pvalues(X, y):
    if 'const' not in X.columns:
        new_x = sm.add_constant(X.columns)
    else:
        new_x = X
    
    vars_used = list(X.columns)
    vars_remaining = list(set(vars_used) - set(['const']))
    model = sm.OLS(y, new_x).fit()
    best_r2 = model.rsquared_adj

    for i in range(len(vars_remaining)):
        var_greatest_pvalue = model.pvalues.sort_values(ascending=False).keys()[0]
        vars_used.remove(var_greatest_pvalue)

        model = sm.OLS(y, new_x[vars_used]).fit()
        r2 = model.rsquared_adj

        if r2 > best_r2:
            continue
        else:
            break
    
    return sm.OLS(y, new_x[vars_used]).fit(), vars_used


df = pd.read_csv(r'C:\Users\tomas\Desktop\modelosSimulacion\data\boston.csv')
set_vars_expl = df.drop('MEDV', axis=1)
set_vars_expl = sm.add_constant(set_vars_expl)
var_obj = df['MEDV']

model, vars_used = bss_pvalues(set_vars_expl, var_obj)
y_pred = model.predict(set_vars_expl[vars_used])

print('Variables utilizadas en el modelo: \n' + str(vars_used))
print('Coeficientes: \n' + str(model.params))
print('mse: ' + str(mse(y_pred, var_obj)))