from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
import preprocessing as pr
import features
'''
    final model pipeline
'''



built_num_pipeline = pr.build_num_pipeline()
built_cat_pipeline = pr.build_cat_pipeline()

predefined_num_cols, predefined_cat_cols = features.get_num_col_features()

def build_final_pipeline(num_pipeline=built_num_pipeline, cat_pipeline=built_cat_pipeline, num_cols=predefined_num_cols, cat_cols=predefined_cat_cols ):

    prep = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, num_cols),
            ('cat', cat_pipeline, cat_cols)
        ]
    )

    final_pipeline = Pipeline([
        ('negotiation_type_checker', FunctionTransformer(pr.fix_operation_type)),
        ('prep', prep),
        ('model', Ridge())
    ])

    return final_pipeline