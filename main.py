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

#CORRIGIR ESTE MAIN.PY

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
model = build_final_pipeline(num_pipeline, cat_pipeline, num_cols, cat_cols, degree_eq=1)

#train test split do dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.09)

#treinamento do modelo com x e y train
model.fit(x_train, y_train)

#avaliação do modelo pós treino
r2, mse, c_val = evaluate_model(model, x_train, x_test, y_train, y_test, k_num=100)
print(f'rmse: {np.sqrt(mse)}')
print(f'r2: {r2}')
print(f'validação cruzada: {np.mean(c_val)}')

#exemplo de dataset para predição
predict_dict = {
    'Condo':[1000],
    'Size':[55],
    'Rooms':[2],
    'Toilets':[2],
    'Suites':[1],
    'Parking':[1],
    'Elevator':[1],
    'Furnished':[1],
    'Swimming Pool':[1],
    'New':[1],
    'Negotiation Type':['rent'],
    'Property Type':['apartment'],
    'Latitude':[-23.482119],
    'Longitude':[-46.601769]
}

#passando para dataframe para melhor reconhecimento pelo modelo para predict
pred_df = pd.DataFrame(predict_dict)

#decode para decodificar o modelo e corrigir com a taxa acumulada de aumento
dec_value = decode_pred(model.predict(pred_df), correction_rate=1.475619, y_log_scale=True)

print(dec_value)

dump_model(model, name='pred_rent_model.joblib')

