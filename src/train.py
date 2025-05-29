from pipeline import build_final_pipeline

'''
    model training based upon a given dataset
'''

def train_model(model, x_train, y_train):
    model.fit(x_train, y_train)
