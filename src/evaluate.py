from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
'''
    evaluate the model predictions
'''

def evaluate_model(model, x_train, y_train, y_test, y_pred, k_num):
    MSError = mean_squared_error(y_pred, y_test)
    cross_validation_score = cross_val_score(model, x_train, y_train, cv=k_num)

    return MSError, cross_validation_score


