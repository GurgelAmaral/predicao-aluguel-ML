import joblib as jb
import sys
sys.path.append('src')
import pandas as pd
from decode import decode_pred

model = jb.load('pred_rent_model.joblib')

predict_dict = {
    'Condo':[500],
    'Size':[35],
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
    'Latitude':[-23.482119],
    'Longitude':[-46.601769]
}

pred_df = pd.DataFrame(predict_dict)
pred_value = decode_pred(model.predict(pred_df), correction_rate=1.47519, y_log_scale=True)

print(pred_value)
