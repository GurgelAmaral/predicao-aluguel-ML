import sys
import pandas as pd
from data.load_data import load_data
from src.features import get_num_col_features
from src.preprocessing import build_cat_pipeline, build_num_pipeline, fix_symmetry
from src.pipeline import build_final_pipeline
from sklearn.model_selection import train_test_split
from src.evaluate import evaluate_model
from sklearn.preprocessing import StandardScaler
import numpy as np
from src.dump import dump_model
from src.decode import decode_pred
from src.tuning import tuned_model



#carregamento dos dados
df, x, y = load_data(path='data\sao-paulo-properties-april-2019.csv',
                     target_col='Price',
                     drop_cat_cols=['District'], 
                     special_case_query='`Negotiation Type` == "rent"',
                     target_log_scale=True)

#criação dos pipelines
num_pipeline = build_num_pipeline()
cat_pipeline = build_cat_pipeline()

fix_symmetry(df, fixable_cols=['Condo', 'Size'])

#captura das features (colunas) para uso no pipeline final
num_cols, cat_cols = get_num_col_features(df)

#criação do pipeline com o modelo final + column transformer dos pipelines numéricos e categóricos
model = build_final_pipeline(num_pipeline, cat_pipeline, num_cols, cat_cols, num_estimators=10)

#train test split do dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.09)

print('Construção modelo base [CHECK]')

#procura pelo melhor modelo de acordo com os hiperparâmetros de tuning.py
new_tuned_model = tuned_model(x_train, y_train, model)

r2, mse, c_val = evaluate_model(new_tuned_model, x_train, x_test, y_train, y_test, k_num=30)
print('Avaliações métricas do melhor modelo:')
print(f'rmse: {np.sqrt(mse)}')
print(f'r2: {r2}')
print(f'validação cruzada: {np.mean(c_val)}')

#armazena modelo em src para uso posterior
try:
    print('-'*128)
    print('salvando e exportando o modelo . . .')
    dump_model(new_tuned_model, name='pred_rent_model.joblib')
    print('Modelo salvo e exportando com sucesso como "pred_rent_model.joblib"')
except Exception as e:
    print(f'Não foi possível salvar e exportar o modelo. Tente novamente | {e}')

