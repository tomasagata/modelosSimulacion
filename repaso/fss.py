# A traves de la base de datos boston, encontrar que variables son las mas adecuadas para determinar
# un modelo de regresion lineal multiple utilizando el algoritmo forward stepwise selection. 
# Imprimir las variables utilizadas por fss, sus coeficientes y el mse

import pandas as pd
import numpy as np
import statsmodels.api as sm

def mse(y_pred: pd.Series, y_real: pd.Series):
    return (np.power(y_pred - y_real, 2)).mean()

def fss(X, y):
    if 'const' not in X.columns:
        new_x = sm.add_constant(X)
    else:
        new_x = X

    vars_used = ['const']
    max_r2 = 0
    vars_remaining = list(set(new_x.columns) - set(vars_used))
    for i in range(len(new_x.columns)):

        best_var = ''
        for var_name in vars_remaining:
            vars_used.append(var_name)
            model = sm.OLS(y, new_x[vars_used]).fit()
            r2 = model.rsquared_adj

            if r2 > max_r2:
                max_r2 = r2
                best_var = var_name
            vars_used.remove(var_name)
        
        if best_var != '':
            vars_used.append(best_var)
        else:
            break
        vars_remaining = list(set(new_x.columns) - set(vars_used))
    
    return sm.OLS(y, new_x[vars_used]).fit(), vars_used



df = pd.read_csv(r'C:\Users\tomas\Desktop\modelosSimulacion\data\boston.csv')
set_vars_expl = df.drop(['MEDV'], axis=1)
set_vars_expl = sm.add_constant(set_vars_expl)
var_obj = df['MEDV']

model, var_name_arr = fss(set_vars_expl, var_obj)
y_pred = model.predict(set_vars_expl[var_name_arr])
print('Variables determinadas por fss: ' + str(var_name_arr))
print('Coeficientes: \n' + str(model.params))
print('mse: ' + str(mse(y_pred, var_obj)))
