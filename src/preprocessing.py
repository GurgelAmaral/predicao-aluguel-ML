from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import sys
sys.path.append('data')
from load_data import load_data

'''
    numeric and cat pipelines
'''

def define_train_test(df=load_data(), y_target_column_name='Price'):
    df_x = df.drop(columns=[y_target_column_name])
    df_y = df[y_target_column_name]

    x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2)
    
    train_df = x_train.copy()
    train_df[y_target_column_name] = y_train
    
    test_df = x_test.copy()
    test_df[y_target_column_name] = y_test
    
    train_df.to_csv("src/train.csv", index=False)
    test_df.to_csv("src/test.csv", index=False)

define_train_test()

#arruma a coluna Negotiation Type caso não seja operação de compra nem de venda, baseado no preço
'''def fix_operation_type(df=load_data()):
    df = df.copy()
    condition = df['Negotiation Type'].isna() and (df['Price'] < df['Price'].max())
    df.loc[condition, 'Negotiation Type'] = 'rent'
    df['Negotiation Type'].fillna("sell", inplace=True)
    return df'''

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
