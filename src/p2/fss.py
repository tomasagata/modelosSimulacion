import pandas as pd
import statsmodels.api as sm


def fss(X, y):
    max_r2 = float(0)
    model_variables = ['const']
    for i in range(1, len(X.columns)):
        remaining_variables = list(set(X.columns) - set(model_variables))
        best_iter_var = ''

        for name_var_expl in remaining_variables:
            model_variables.append(name_var_expl)
            new_model = sm.OLS(y, X[model_variables]).fit()
            if new_model.rsquared_adj > max_r2:
                best_iter_var = name_var_expl
                max_r2 = new_model.rsquared_adj
            model_variables.pop()

        if best_iter_var == '':
            break

        model_variables.append(best_iter_var)
    
    return sm.OLS(y, X[model_variables]).fit()


if __name__ == '__main__':
    boston = pd.read_csv(r"/Users/kpapiccio/PycharmProjects/modelosSimulacion/data/boston.csv")
    set_vars_expl = boston.drop(['MEDV'], axis=1)
    set_vars_expl = sm.add_constant(set_vars_expl)
    var_obj = boston[['MEDV']]

    model = fss(set_vars_expl, var_obj)
    print(model.summary())
