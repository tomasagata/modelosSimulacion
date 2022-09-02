import pandas as pd
from matplotlib import pyplot as plt
import statsmodels.api as sm


def plotear_con_var_obj(var_explicativas, var_obj):
    for var_expl_name, var_expl_values in var_explicativas.items():
        plt.scatter(var_expl_values, var_obj)
        plt.ylabel(var_obj.columns[0])
        plt.xlabel(var_expl_name)
        plt.show()


def get_all_values_in_arr_except_for(arr, blacklisted_items):
    new_arr = []
    for item in arr:
        if item not in blacklisted_items:
            new_arr.append(item)
    return new_arr


def forward_stepwise_selection(var_explicativas, var_obj):
    var_explicativas = sm.add_constant(var_explicativas)
    variables = ['const']
    r2_data = {
        "max_r2_value": 0,
        "max_r2_variables": ['const'],
        "max_r2_in_iteration_value": 0,
        "max_r2_in_iteration_variables": ['const']
    }
    for iteration in range(len(var_explicativas) - 1):
        for curr_var in var_explicativas.columns.drop(variables, 1):
            current_variables = variables.copy()
            current_variables.append(curr_var)
            regr = regresion(var_explicativas[current_variables], var_obj)

            if regr.rsquared_adj > r2_data["max_r2_in_iteration_value"]:
                r2_data["max_r2_in_iteration_value"] = regr.rsquared_adj
                r2_data["max_r2_in_iteration_variables"] = current_variables

        if r2_data["max_r2_in_iteration_value"] > r2_data["max_r2_value"]:
            r2_data["max_r2_value"] = r2_data["max_r2_in_iteration_value"]
            r2_data["max_r2_variables"] = r2_data["max_r2_in_iteration_variables"]

        variables = r2_data["max_r2_in_iteration_variables"]

    return regresion(var_explicativas[r2_data["max_r2_variables"]], var_obj)

# def backward_stepwise_selection(var_explicativas, var_obj):
#     var_explicativas = sm.add_constant(var_explicativas)
#     pass

def regresion(var_explicativas, var_obj):
    temp = sm.OLS(var_obj, var_explicativas).fit()
    return temp


if __name__ == '__main__':
    # Cargamos los dataframes
    startups = pd.read_csv('50_Startups.csv')
    vars_explicativas = startups.drop(columns=['Profit'])
    var_objetiva = startups[['Profit']]

    # A vars_explicativas ya le agrego la constante
    vars_explicativas = sm.add_constant(vars_explicativas)

    # Para poder hacer una regresión lineal con OLS, debemos
    # Cambiar las variables categóricas a numéricas
    # Nueva York = 0, California = 1, Florida = 2
    vars_explicativas['State'].replace(['New York', 'California', 'Florida'],
                                       [0, 1, 2], inplace=True)

    plotear_con_var_obj(vars_explicativas, var_objetiva)

    regr = regresion(var_objetiva, vars_explicativas)
    # print(regr.summary())

    print(forward_stepwise_selection(vars_explicativas, var_objetiva).summary())
