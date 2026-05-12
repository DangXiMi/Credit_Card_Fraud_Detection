from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

def get_pipeline(params, imbalance_ratio):
    clf_params = {k[12:]: v for k, v in params.items() if k.startswith('classifier__')}
    pipeline = Pipeline([
        ('classifier', XGBClassifier(
            **clf_params,
            scale_pos_weight=imbalance_ratio,
            eval_metric='logloss',
            random_state=42
        ))
    ])
    return pipeline


def build_pipeline(scale_pos_weight: float) -> Pipeline:
    model = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        max_delta_step=1,
        eval_metric='logloss',
        random_state=42
    )
    return Pipeline([('classifier', model)])