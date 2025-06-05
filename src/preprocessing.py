from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from scipy import stats
'''
    numeric and cat pipelines
'''

#arruma a coluna Negotiation Type caso não seja operação de compra nem de venda, baseado no preço
'''def fix_operation_type(df=load_data()):
    df = df.copy()
    condition = df['Negotiation Type'].isna() and (df['Price'] < df['Price'].max())
    df.loc[condition, 'Negotiation Type'] = 'rent'
    df['Negotiation Type'].fillna("sell", inplace=True)
    return df'''

def fix_symmetry(df, fixable_cols=None):
    if fixable_cols is not None:
        for col in fixable_cols:
            df[col], _ = stats.boxcox(df[col] + 1e-6)

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
