'''
    columns and types selection
'''

#função que captura as colunas numéricas e categóricas
def get_num_col_features(df):
    return df.select_dtypes(include=["number"]).columns, df.select_dtypes(include=["object", "string"]).columns