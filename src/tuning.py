from sklearn.model_selection import GridSearchCV
import time

'''
    find best hyperparams for the model
'''

#grid de hyperparams | pode ser modificado
hyp_param_grid = {
    'model__n_estimators':[50, 100, 200, 300, 400, 500],
    'model__max_depth':[None, 5, 10, 20, 30, 40]
}

#função para retornar o melhor modelo com base no grid de hiperparams
def tuned_model(x_train, y_train, model, cv_k=5, scoring_method='r2'):
    print('Preparação para busca de melhor modelo [CHECK]')
    print('-'*128)
    grid = GridSearchCV(estimator=model ,param_grid=hyp_param_grid, cv=cv_k, scoring=scoring_method, n_jobs=-1)

    try:
        print('Buscando e treinando melhor modelo . . .')
        start = time.time()
        grid.fit(x_train, y_train)
        finish = time.time()
        print(f'Tempo de busca e treino: {finish - start} s')
        return grid.best_estimator_
    
    except Exception as e:
        print(f'não foi possível retornar o melhor estimador | {e}')