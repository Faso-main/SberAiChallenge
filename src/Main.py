import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

def try_lightgbm():
    """Попытка использовать LightGBM с правильным API"""
    print("Trying LightGBM with correct API...")
    
    # Загрузка данных
    train_df = pd.read_csv('src/ctr_train.csv', nrows=100000)
    test_df = pd.read_csv('src/ctr_test.csv')
    
    # Простая предобработка
    def simple_preprocess(df):
        numeric_cols = ['C1', 'banner_pos', 'device_type', 'device_conn_type', 
                       'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21']
        return df[numeric_cols]
    
    X = simple_preprocess(train_df)
    y = train_df['click']
    X_test = simple_preprocess(test_df)
    
    # Разделение
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    try:
        # Создаем datasets для LightGBM
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # Параметры
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1
        }
        
        # Обучение
        model = lgb.train(params,
                         train_data,
                         num_boost_round=1000,
                         valid_sets=[val_data],
                         callbacks=[lgb.early_stopping(stopping_rounds=50)])
        
        # Предсказание
        val_pred = model.predict(X_val)
        val_auc = roc_auc_score(y_val, val_pred)
        print(f'LightGBM Validation AUC: {val_auc:.5f}')
        
        return model, val_auc
        
    except Exception as e:
        print(f"LightGBM failed: {e}")
        return None, 0

# Запуск
model, lgb_auc = try_lightgbm()
if lgb_auc > 0.7:
    print("Using LightGBM model")
    # Сделать предсказания на тесте и создать сабмит
else:
    print("Using RandomForest model")
    # Использовать RandomForest подход