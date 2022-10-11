# En este script, lo que hacemos es obtener las 3 variables
# explicativas que mejor se ajustan al modelo y, a partir de
# ellas, armamos un nuevo modelo usando estas 3

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def bestN(dict, n):
    keys = list(dict.keys())
    values = list(dict.values())

    best_scores = {
        'values': [],
        'keys': []
    }
    if len(values) > 0:
        for i in range(n):
            best_scores['values'] = [values[0]] * n
            best_scores['keys'] = [keys[0]] * n
    else:
        return best_scores 

    for i in range(1, len(dict)):
        for j in range(n):
            if values[i] > best_scores['values'][j]:
                for l in range(n-1, j, -1):
                    best_scores['values'][l] = best_scores['values'][l-1]
                    best_scores['keys'][l] = best_scores['keys'][l-1]
                best_scores['values'][j] = values[i]
                best_scores['keys'][j] = keys[i]
                break

    return best_scores


if __name__ == '__main__':
    boston = pd.read_csv(r"C:\Users\tomas\Desktop\modelosSimulacion\data\boston.csv")
    set_var_expl = boston.drop(['MEDV'], axis=1)
    var_obj = boston['MEDV']
    
    r2_scores = {}
    for name_var_expl in set_var_expl.columns:
        var_expl = set_var_expl[[name_var_expl]]

        model = LinearRegression()
        model.fit(var_expl, var_obj)

        prediccion = model.predict(var_expl)
        r2_scores[name_var_expl] = r2_score(var_obj, prediccion)

    best_scores = bestN(r2_scores, 3)

    model = LinearRegression()
    model.fit(set_var_expl[best_scores['keys']], var_obj)
    
    prediccion = model.predict(set_var_expl[best_scores['keys']])
    print(best_scores)
    print("El modelo en conjunto con las 3 variables da un r2 de " + str(r2_score(var_obj, prediccion)))