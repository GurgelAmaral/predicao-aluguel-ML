import sys
sys.path.append('data')
'''
    columns and types selection
'''
import load_data as ld

#função que captura as colunas numéricas e categóricas

def get_num_col_features(df=ld.load_data()):
    return df.select_dtypes(include=["number"]).columns.to_list(), df.select_dtypes(include=["object", "string"]).columns.to_list()
