from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
'''
    final model pipeline
'''


def build_final_pipeline(num_pipeline, cat_pipeline, num_cols, cat_cols):

    prep = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, num_cols),
            ('cat', cat_pipeline, cat_cols)
        ]
    )

    final_pipeline = Pipeline([
        #('negotiation_type_checker', FunctionTransformer(pr.fix_operation_type)),
        ('prep', prep),
        ('model', LinearRegression())
    ])

    return final_pipeline