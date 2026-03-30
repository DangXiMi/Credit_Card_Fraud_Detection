import joblib
import logging
import numpy as np # Added numpy
import preprocessing as pp
from pipeline import get_pipeline
from config import SETTINGS
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import average_precision_score, precision_recall_curve

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def run_training():
    logger.info("Loading data...")
    df = pp.load_data(SETTINGS['paths']['raw_data'])
    df = pp.clean_raw_data(df)
    
    X_train, X_val, X_test, y_train, y_val, y_test = pp.split_time_chronological(
        df, 
        train_size=SETTINGS['train_params']['train_size'],
        val_size=SETTINGS['train_params']['val_size']
    )
    
    imbalance_ratio = (y_train == 0).sum() / (y_train == 1).sum()
    tscv = TimeSeriesSplit(n_splits=SETTINGS['train_params']['cv_splits'])
    
    base_pipe = get_pipeline(params={}, imbalance_ratio=imbalance_ratio)
    param_grid = SETTINGS['search_space']

    search = RandomizedSearchCV(
        estimator=base_pipe,
        param_distributions=param_grid,
        n_iter=SETTINGS['train_params']['n_iter'],
        scoring='average_precision',
        cv=tscv,
        n_jobs=-1,
        verbose=1,
        random_state=42
    )
    
    logger.info("Starting Search...")
    search.fit(X_train, y_train)
    
    logger.info(f"Best Hyperparameters Found: {search.best_params_}")
    
    joblib.dump(search.best_estimator_, SETTINGS['paths']['model_output'])
    
    y_probs = search.best_estimator_.predict_proba(X_val)[:, 1]
    
    precisions, recalls, thresholds = precision_recall_curve(y_val, y_probs)
    
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    
    logger.info(f"Val PR-AUC: {average_precision_score(y_val, y_probs):.4f}")
    logger.info(f"Optimal Threshold for F1: {best_threshold:.4f} (Validation F1: {best_f1:.4f})")

if __name__ == "__main__":
    run_training()