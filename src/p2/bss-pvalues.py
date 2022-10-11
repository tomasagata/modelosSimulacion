import pandas as pd
import numpy as np
import statsmodels.api as sm


def bss_pvalues(X, y):
    model_variables = list(X.columns)
    new_model = sm.OLS(y, X[model_variables]).fit()
    max_r2 = new_model.rsquared_adj
    for i in range(1, len(X.columns)):

        pvalues_arr = new_model.pvalues.drop('const')
        max_pvalue_var_name = pvalues_arr.keys()[pvalues_arr.argmax()]

        model_variables.remove(max_pvalue_var_name)
        new_model = sm.OLS(y, X[model_variables]).fit()
        if new_model.rsquared_adj > max_r2:
            max_r2 = new_model.rsquared_adj
        else:
            model_variables.append(max_pvalue_var_name)
            return sm.OLS(y, X[model_variables]).fit()

if __name__ == '__main__':
    boston = pd.read_csv(r"/Users/kpapiccio/PycharmProjects/modelosSimulacion/data/boston.csv")
    set_vars_expl = boston.drop(['MEDV'], axis=1)
    set_vars_expl = sm.add_constant(set_vars_expl)
    var_obj = boston[['MEDV']]

    model = bss_pvalues(set_vars_expl, var_obj)
    print(model.summary())

