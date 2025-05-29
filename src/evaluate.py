from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
'''
    evaluate the model predictions
'''

def evaluate_model(model, x_train, y_train, y_test, y_pred, k_num):
    MSError = mean_squared_error(y_test, y_pred)
    cross_validation_score = cross_val_score(model, x_train, y_train, cv=k_num)
    r2 = r2_score(y_test, y_pred)
    return r2, MSError, cross_validation_score


