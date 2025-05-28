from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

'''
    numeric and cat pipelines
'''

#arruma a coluna operation type caso não seja operação de compra nem de venda, baseado no preço
def fix_operation_type(df):
    df = df.copy()
    condition = df['operation type'].isna() & (df['price'] < df['price'].max())
    df.loc[condition, 'operation type'] = 'rent'
    df['operation type'].fillna("sell", inplace=True)
    return df

#construi o pipeline das variáveis numéricas
def build_num_pipeline():
    num_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])
    return num_pipeline

#construi pipeline das variáveis categóricas
def build_cat_pipeline():
    cat_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])
    return cat_pipeline
