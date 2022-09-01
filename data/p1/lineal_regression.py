from sklearn import linear_model
from sklearn.metrics import r2_score
import pandas as pd
import matplotlib.pyplot as plt


def read_and_show_head_with_columns(filename, columns_to_show):
    data_complete = pd.read_csv(filename)
    data_selected = data_complete[columns_to_show]
    data_selected.head()
    return data_selected


if __name__ == '__main__':
    vars_explicativas = read_and_show_head_with_columns(
        "boston.csv",
        ['CRIM',
         'ZN',
         'INDUS',
         'CHAS',
         'NOX',
         'RM',
         'AGE',
         'DIS',
         'RAD',
         'TAX',
         'PTRATIO',
         'B',
         'LSTAT']
    )
    var_objetiva = read_and_show_head_with_columns("boston.csv", 'MEDV')

    regr = linear_model.LinearRegression()
    var_expl = vars_explicativas[['RM']]
    regr.fit(var_expl, var_objetiva)
    print({
        'beta1': regr.coef_,
        'beta0': regr.intercept_
    })
    prediccion = regr.predict(var_expl)
    plt.scatter(var_expl, var_objetiva)
    plt.plot(var_expl, prediccion, color='r')
    plt.xlabel('LSTAT')
    plt.ylabel('MEDV')
    print(r2_score(var_objetiva, prediccion))
    plt.show()
