from joblib import dump

#faz o dump do modelo já treinado para ser reutilizado
def dump_model(model, name='model_dump.joblib'):
    try:
        dump(model, name)
    except Exception as e:
        print(f'Não foi possível salvar o modelo | {e}')