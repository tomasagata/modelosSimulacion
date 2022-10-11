import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression


def factorize(dataframe, var_name_arr):
    new_df = dataframe
    for var_name in var_name_arr:
        var = pd.factorize(new_df[var_name])
        index = new_df.columns.get_loc(var_name)
        new_df = new_df.drop([var_name], axis=1)
        new_df.insert(index, var_name, var[0])
    return new_df

def k_cross_validation_score(X: pd.DataFrame, y: pd.Series, k = 5, threshold = 0.5):
    scores = []
    for test_segment in range(k):
        _range_start = int(np.floor((test_segment/k) * len(X)))
        _range_end = int(np.floor(((test_segment+1)/k) * len(X)))
        train_X = X.drop(np.arange(_range_start, _range_end), axis=0)
        train_y = list(y.drop(np.arange(_range_start, _range_end), axis=0))

        model_k = LogisticRegression().fit(train_X, train_y)
        scores.append(logistic_regression_score(model_k, train_X, train_y, threshold))
    return np.mean(scores)

def logistic_regression_score(model, X, y, threshold = 0.5):
    y_proba = model.predict_proba(X)
    y_size = len(y_proba)
    hits = 0
    for i in range(y_size):
        y_pred = 0
        if y_proba[i][1] > threshold:
            y_pred = 1
        
        if y[i] == y_pred:
            hits += 1
    return hits/y_size

def confusion_matrix(model, X, y, threshold = 0.5):
    y_proba = model.predict_proba(X)
    y_size = len(y_proba)
    tp = 0; fp = 0; fn = 0; tn = 0
    for i in range(y_size):
        y_pred = 0
        if y_proba[i][1] > threshold:
            y_pred = 1
        
        if y_pred == 1 and y[i] == 1:
            tp += 1
        elif y_pred == 1 and y[i] == 0:
            fp += 1
        elif y_pred == 0 and y[i] == 1:
            fn += 1
        elif y_pred == 0 and y[i] == 0:
            tn += 1
    return tp, fp, fn, tn

def expected_benefits(model, X, y, threshold = 0.5, tp_benefits = 1, fp_benefits = -1, fn_benefits = -1, tn_benefits = 1):
    tp, fp, fn, tn = confusion_matrix(model, X, y, threshold)
    return tp_benefits * tp + fp_benefits * fp + fn_benefits * fn + tn_benefits * tn

def logistic_regression_mse(model, X, y, threshold = 0.5):
    y_proba = model.predict_proba(X)
    y_size = len(y_proba)
    predicted = []
    for i in range(y_size):
        y_pred = 0
        if y_proba[i][1] > threshold:
            y_pred = 1
        
        predicted.append(y_pred)
    return (np.square(predicted - y)).mean()

if __name__ == '__main__':
    df = pd.read_csv(r'/Users/kpapiccio/PycharmProjects/modelosSimulacion/data/Churn_Modelling.csv')
    df = df.drop(['RowNumber','CustomerId','Surname'], axis=1)
    set_vars_expl = df.drop(['Exited'], axis=1)
    set_vars_expl = factorize(set_vars_expl, ['Geography', 'Gender'])
    var_obj = df['Exited']

    model = LogisticRegression().fit(set_vars_expl, var_obj)
    for threshold in np.arange(0, 1, 0.1):
        print('--------------------------------------')
        print('Threshold: ' + str(round(threshold, 1)))
        print('k-cross-validation: ' + str(k_cross_validation_score(set_vars_expl, var_obj, 5, threshold)))
        print('Estimated Benefits: ' + str(expected_benefits(model, set_vars_expl, var_obj, threshold)))
        print('Mean Squared Error: ' + str(logistic_regression_mse(model, set_vars_expl, var_obj, threshold)))