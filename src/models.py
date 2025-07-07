from decode import decode_pred
import joblib as jb
import pandas as pd

model = jb.load('pred_rent_model.joblib')

#exemplo de dataset para predição
predict_dict = {
    'Condo':[1],
    'Size':[30],
    'Rooms':[1],
    'Toilets':[1],
    'Suites':[0],
    'Parking':[0],
    'Elevator':[0],
    'Furnished':[0],
    'Swimming Pool':[0],
    'New':[0],
    'Negotiation Type':['rent'],
    'Property Type':['apartment'],
    'Latitude':[-23.481035],
    'Longitude':[-46.570452]
}

#passando para dataframe para melhor reconhecimento pelo modelo para predict
pred_df = pd.DataFrame(predict_dict)

prediction = model.predict(pred_df)
print(f'Aluguel aproximado calculado do imóvel: R$ {decode_pred(predicted_value=prediction, y_log_scale=True)}')