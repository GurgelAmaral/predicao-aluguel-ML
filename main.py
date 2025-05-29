import sys
import pandas as pd
from data.load_data import load_data
from src.features import get_num_col_features
from src.preprocessing import build_cat_pipeline, build_num_pipeline
from src.pipeline import build_final_pipeline
from sklearn.model_selection import train_test_split
from src.evaluate import evaluate_model
from sklearn.preprocessing import StandardScaler
import numpy as np
#carregamento dos dados
df = load_data()
df = df.drop(columns=['District'])
df = df.query('`Negotiation Type` == "rent"')

#aquisição dos nomes das colunas numéricas e categóricas para o column transformer
num_cols, cat_cols = get_num_col_features(df)
num_cols.remove('Price')

#criação dos pipelines
num_pipeline = build_num_pipeline()
cat_pipeline = build_cat_pipeline()

#criação do pipeline com o modelo final + column transformer dos pipelines numéricos e categóricos
model = build_final_pipeline(num_pipeline, cat_pipeline, num_cols, cat_cols)

#train test split do dataset
x = df.drop(columns=['Price'])
scaler = StandardScaler()
y = np.log(df['Price'])
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15)

#treinamento do modelo com x e y train
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

#avaliação do modelo pós treino
r2, mse, c_val = evaluate_model(model, x_train, y_train, y_test, y_pred=y_pred, k_num=30)
print(f'rmse: {np.sqrt(mse)}')
print(f'r2: {r2}')
print(f'validação cruzada: {np.mean(c_val)}')

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

pred_df = pd.DataFrame(predict_dict)

print(np.exp(model.predict(pred_df)))
