'''
    load csv info as a pandas dataframe
'''

import pandas as pd
import numpy as np


#carrega o df do path e retorna x e y já limpos
def load_data(path, drop_cat_cols, drop_num_cols, target_col, special_case_query, target_log_scale=False):
    #prepara o df dropando as colunas que não se aplicam ao problema
    df = pd.read_csv(path)
    df = df.drop(columns=drop_cat_cols)
    df = df.drop(columns=drop_num_cols)
    df = df.query(special_case_query)

    #definindo x a partir de df, retirando a coluna target
    x = df.drop(columns=[target_col])

    #definição de y a partir de df pela coluna target e aplicando escala logarítmica se houver condição
    if target_log_scale:
        y = np.log(df[target_col])
    else:
        y = df[target_col]

    #remoção da coluna target para bom funcionamento do pipeline caso já não tenha sido removida
    if (target_col not in drop_num_cols and target_col not in drop_cat_cols):
        df = df.drop(columns=[target_col])

    return df, x, y




