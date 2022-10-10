import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import r2_score


if __name__ == '__main__':
    boston = pd.read_csv(r"C:\Users\tomas\Desktop\modelosSimulacion\data\boston.csv")
    set_var_expl = boston.drop(['MEDV'], axis=1)
    var_obj = boston[['MEDV']]

    r2_scores = {}
    for name_var_expl in set_var_expl.columns:
        var_expl = set_var_expl[[name_var_expl]]

        model = linear_model.LinearRegression()
        model.fit(var_expl, var_obj)

        prediccion = model.predict(var_expl)
        r2 = r2_score(var_obj, prediccion)
        r2_scores[name_var_expl] = r2

        plt.scatter(var_expl.values, var_obj, c='r')
        plt.plot(var_expl.values, prediccion, c='g')
        plt.legend(['R2 = ' + str(r2)], loc="upper left")
        plt.xlabel(name_var_expl)
        plt.ylabel('MEDV')
        plt.show()
    
    r2_max = 0
    max_label = ''
    for name_var_expl in r2_scores:
        if r2_scores[name_var_expl] > r2_max:
            r2_max = r2_scores[name_var_expl]
            max_label = name_var_expl
    
    print("La mejor predicción se dio con la variable " + max_label + " con un valor de " + str(r2_max))
