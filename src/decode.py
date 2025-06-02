import numpy as np

'''
    method to decode the prediction and correct it when needed
'''

def decode_pred(predicted_value, correction_rate, y_log_scale=False):
    #transforma o valor predito para a escala original se há escala logarítmica aplicada
    if y_log_scale:
        predicted_value = np.exp(predicted_value)

    #calcula o valor ajustado acumulado de 2019 até o ano vigente
    correct_real_value = predicted_value * correction_rate
    
    return correct_real_value