import pandas as pd
import statsmodels.api as sm


def bss(X, y):
    model_variables = X.columns
    new_model = sm.OLS(y, X[model_variables]).fit()
    max_r2 = model.rsquared_adj
    for i in range(1, len(X.columns)):
        remaining_variables = list(set(X.columns) - set(model_variables) - set(['const']))
        best_iter_var = ''

        for name_var_expl in remaining_variables:
            model_variables.remove(name_var_expl)
            new_model = sm.OLS(y, X[model_variables]).fit()
            if new_model.rsquared_adj > max_r2:
                best_iter_var = name_var_expl
                max_r2 = new_model.rsquared_adj
            model_variables.append(name_var_expl)
        
        if best_iter_var == '':
            break
        
        model_variables.remove(name_var_expl)

    return sm.OLS(y, X[model_variables]).fit()

if __name__ == '__main__':
    boston = pd.read_csv(r"/Users/kpapiccio/PycharmProjects/modelosSimulacion/data/boston.csv")
    set_vars_expl = boston.drop(['MEDV'], axis=1)
    set_vars_expl = sm.add_constant(set_vars_expl)
    var_obj = boston[['MEDV']]

    model = bss(set_vars_expl, var_obj)
    print(model.summary())

