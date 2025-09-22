import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_auc_score
import catboost as cb
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
import gc
import warnings
warnings.filterwarnings('ignore')

# Увеличиваем до 3-5 млн строк
SAMPLE_SIZE = 3000000

def expert_feature_engineering(df, is_train=True):
    """Экспертный feature engineering"""
    # Детальная обработка времени
    if 'hour' in df.columns:
        df['hour'] = df['hour'].astype(str)
        hour_dt = pd.to_datetime(df['hour'].str[:6], format='%y%m%d')
        
        # Временные признаки
        df['hour_day'] = hour_dt.dt.day
        df['hour_month'] = hour_dt.dt.month
        df['hour_weekday'] = hour_dt.dt.weekday
        df['hour_hour'] = pd.to_numeric(df['hour'].str[6:])
        df['hour_is_weekend'] = (df['hour_weekday'] >= 5).astype(int)
        df['hour_is_night'] = ((df['hour_hour'] >= 22) | (df['hour_hour'] <= 6)).astype(int)
        df['hour_is_morning'] = ((df['hour_hour'] >= 6) & (df['hour_hour'] <= 10)).astype(int)
        df['hour_is_evening'] = ((df['hour_hour'] >= 18) & (df['hour_hour'] <= 22)).astype(int)
        
        df.drop(columns=['hour'], inplace=True)
    
    # Сложные комбинации признаков
    if all(col in df.columns for col in ['site_id', 'app_id', 'device_type']):
        df['site_app_device'] = df['site_id'] + '_' + df['app_id'] + '_' + df['device_type'].astype(str)
    
    if all(col in df.columns for col in ['site_category', 'app_category']):
        df['site_app_category'] = df['site_category'] + '_' + df['app_category']
    
    # Статистические признаки из C-признаков
    if all(col in df.columns for col in ['C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21']):
        c_cols = ['C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21']
        df['c_mean'] = df[c_cols].mean(axis=1)
        df['c_std'] = df[c_cols].std(axis=1)
        df['c_sum'] = df[c_cols].sum(axis=1)
        
        # Взаимодействия
        df['C14_C15_ratio'] = df['C14'] / (df['C15'] + 1)
        df['C15_C16_ratio'] = df['C15'] / (df['C16'] + 1)
        df['C17_C18_ratio'] = df['C17'] / (df['C18'] + 1)
        df['C19_C20_ratio'] = df['C19'] / (df['C20'] + 1)
    
    # Бинарные признаки
    if 'banner_pos' in df.columns:
        df['is_banner_pos_0'] = (df['banner_pos'] == 0).astype(int)
        df['is_banner_pos_1'] = (df['banner_pos'] == 1).astype(int)
    
    return df

def create_aggregated_features(train_df, test_df, target='click'):
    """Создание агрегированных признаков"""
    # Агрегации по основным группам
    aggregation_groups = [
        'site_id', 'app_id', 'device_type', 'device_conn_type',
        'site_domain', 'app_domain', 'site_category', 'app_category'
    ]
    
    for group_col in aggregation_groups:
        if group_col in train_df.columns:
            # Статистики по тренировочным данным
            group_stats = train_df.groupby(group_col)[target].agg([
                'mean', 'std', 'count', 'sum', 'median'
            ]).fillna(0)
            
            # Новые имена колонок
            new_cols = [f'{group_col}_{stat}' for stat in group_stats.columns]
            group_stats.columns = new_cols
            
            # Добавляем к данным
            train_df = train_df.merge(group_stats, on=group_col, how='left')
            test_df = test_df.merge(group_stats, on=group_col, how='left')
            
            # Заполняем пропуски в тесте
            for col in new_cols:
                test_df[col].fillna(train_df[col].median(), inplace=True)
    
    return train_df, test_df

def main():
    print("Loading data...")
    train_df = pd.read_csv('task1/src/ctr_train.csv', nrows=SAMPLE_SIZE)
    test_df = pd.read_csv('task1/src/ctr_test.csv')
    
    print("Expert feature engineering...")
    train_df = expert_feature_engineering(train_df)
    test_df = expert_feature_engineering(test_df, is_train=False)
    
    print("Creating aggregated features...")
    train_df, test_df = create_aggregated_features(train_df, test_df)
    
    # Удаляем только самые проблемные колонки
    cols_to_drop = ['id', 'device_id', 'device_ip', 'device_model']
    train_df.drop(columns=[col for col in cols_to_drop if col in train_df.columns], inplace=True)
    test_df.drop(columns=[col for col in cols_to_drop if col in test_df.columns], inplace=True)
    
    # Подготовка данных
    y = train_df['click']
    X = train_df.drop(columns=['click'])
    
    # Определяем категориальные признаки
    cat_features = list(X.select_dtypes(include=['object']).columns)
    print(f"Categorical features: {len(cat_features)}")
    
    # Преобразуем в category
    for col in cat_features:
        X[col] = X[col].astype('category')
        test_df[col] = test_df[col].astype('category')
    
    print("Training expert CatBoost model...")
    
    # Еще более оптимизированные параметры
    model = cb.CatBoostClassifier(
        iterations=4000,           # Увеличили количество итераций
        learning_rate=0.02,        # Уменьшили learning rate
        depth=12,                  # Увеличили глубину
        l2_leaf_reg=3,
        random_strength=0.7,
        bagging_temperature=0.8,
        border_count=254,
        loss_function='Logloss',
        eval_metric='AUC',
        task_type='CPU',
        random_state=42,
        verbose=200,
        early_stopping_rounds=200,  # Увеличили patience
        use_best_model=True,
        nan_mode='Min'             # Обработка NaN
    )
    
    # Разделяем данные
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        cat_features=cat_features
    )
    
    # Оценка
    val_pred = model.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, val_pred)
    print(f'Expert CatBoost Validation AUC: {val_auc:.5f}')
    
    # Кросс-валидация для надежности
    print("Cross-validation...")
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
        
        fold_model = cb.CatBoostClassifier(
            iterations=2000,
            learning_rate=0.02,
            depth=12,
            random_state=42,
            verbose=0
        )
        
        fold_model.fit(X_train_fold, y_train_fold, 
                      cat_features=cat_features, 
                      verbose=0)
        
        fold_pred = fold_model.predict_proba(X_val_fold)[:, 1]
        fold_auc = roc_auc_score(y_val_fold, fold_pred)
        cv_scores.append(fold_auc)
        print(f'Fold {fold+1} AUC: {fold_auc:.5f}')
    
    print(f'Mean CV AUC: {np.mean(cv_scores):.5f} (+/- {np.std(cv_scores):.5f})')
    
    # Анализ важности признаков
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.get_feature_importance()
    }).sort_values('importance', ascending=False)
    
    print("\nTop 15 most important features:")
    print(feature_importance.head(15))
    
    # Предсказание на тесте
    print("Making predictions...")
    test_pred = model.predict_proba(test_df)[:, 1]
    
    # Постобработка предсказаний
    test_pred = np.clip(test_pred, 0.005, 0.995)
    
    # Создание сабмита
    submission = pd.read_csv('task1/src/ctr_sample_submission.csv')
    submission['click'] = test_pred
    submission.to_csv('expert_catboost_submission.csv', index=False)
    print("Expert CatBoost submission created!")
    print(f"Final Validation AUC: {val_auc:.5f}")

if __name__ == "__main__":
    main()