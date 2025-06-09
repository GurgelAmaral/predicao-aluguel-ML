from joblib import dump

#faz o dump do modelo já treinado para ser reutilizado
def dump_model(model, compress_value=1, name='model_dump.joblib'):
    try:
        dump(model, name, compress=compress_value)
    except Exception as e:
        print(f'Não foi possível salvar o modelo | {e}')