# src/pipeline.py
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier

def get_pipeline(params, imbalance_ratio):
    """
    Creates a unified pipeline: 
    1. Scales 'Time' and 'Amount'
    2. Passes V1-V28 through unchanged
    3. Fits XGBoost
    """
    cols_to_scale = ['Hour', 'Amount']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('scaler', RobustScaler(), cols_to_scale)
        ],
        remainder='passthrough' 
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(
            **params,
            scale_pos_weight=imbalance_ratio,
            eval_metric='logloss',
            random_state=42
        ))
    ])
    
    return pipeline