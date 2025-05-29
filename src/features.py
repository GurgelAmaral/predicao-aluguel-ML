'''
    columns and types selection
'''

#função que captura as colunas numéricas e categóricas

def get_num_col_features(df):
    return df.select_dtypes(include=["number"]).columns.to_list(), df.select_dtypes(include=["object", "string"]).columns.to_list()
