# A traves de la base de datos de boston, utilizar un modelo de regresion lineal que pueda predecir el MEDV
# para ello, utilizar el algoritmo bss convencional para encontrar las variables optimas del modelo de datos. 
# Imprimir las variables utilizadas, los coeficientes y finalmente el mse del modelo
import pandas as pd
import numpy as np
import statsmodels.api as sm


def mse(y_pred, y_real):
    return (np.power(y_pred - y_real, 2)).mean()

def bss(X, y):
    if 'const' not in X.columns:
        new_x = sm.add_constant(X)
    else:
        new_x = X
    

    vars_used = list(X.columns)
    vars_remaining = list(set(vars_used) - set(['const']))
    model = sm.OLS(y, new_x[vars_used]).fit()
    max_r2 = model.rsquared_adj

    for i in range(len(X.columns) - 1):
        best_var = ''

        for var_name in vars_remaining:
            vars_used.remove(var_name)

            model = sm.OLS(y, X[vars_used]).fit()
            r2 = model.rsquared_adj
            
            if r2 > max_r2:
                best_var = var_name
                max_r2 = r2
            
            vars_used.append(var_name)
        
        if best_var != '':
            vars_used.remove(best_var)
        else:
            break
        vars_remaining = list(set(vars_used) - set(['const']))

    return sm.OLS(y, new_x[vars_used]).fit(), vars_used



df = pd.read_csv(r'C:\Users\tomas\Desktop\modelosSimulacion\data\boston.csv')
set_vars_expl = df.drop('MEDV', axis=1)
set_vars_expl = sm.add_constant(set_vars_expl)
var_obj = df['MEDV']

model, var_name_arr = bss(set_vars_expl, var_obj)
y_pred = model.predict(set_vars_expl[var_name_arr])

print('Las variables utilizadas fueron: \n' + str(var_name_arr))
print('Coeficientes: \n' + str(model.params))
print('mse: ' + str(mse(y_pred, var_obj)))