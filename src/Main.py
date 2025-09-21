import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import gc
import warnings
warnings.filterwarnings('ignore')

# Уменьшим sample для отладки
SAMPLE_SIZE = 100000

def simple_preprocessing(df):
    """Базовая предобработка данных"""
    # Обработка hour
    if 'hour' in df.columns:
        df['hour'] = df['hour'].astype(str)
        df['hour_weekday'] = pd.to_datetime(df['hour'].str[:6], format='%y%m%d').dt.weekday
        df['hour_hour'] = pd.to_numeric(df['hour'].str[6:])
        df.drop(columns=['hour'], inplace=True)
    
    # Удаляем высококардинальные колонки
    cols_to_drop = ['id', 'device_id', 'device_ip', 'device_model']
    df.drop(columns=[col for col in cols_to_drop if col in df.columns], inplace=True)
    
    return df

def main():
    print("Step 1: Loading data...")
    # Загружаем sample данных
    train_df = pd.read_csv('src/ctr_train.csv', nrows=SAMPLE_SIZE)
    test_df = pd.read_csv('src/ctr_test.csv')
    
    print("Step 2: Preprocessing...")
    # Предобработка
    train_df = simple_preprocessing(train_df)
    test_df = simple_preprocessing(test_df)
    
    # Выделяем целевой признак
    if 'click' in train_df.columns:
        y = train_df['click']
        X = train_df.drop(columns=['click'])
    else:
        raise ValueError("Column 'click' not found in training data")
    
    # Label Encoding для категориальных признаков
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    print(f"Categorical columns: {cat_cols}")
    
    le_dict = {}
    for col in cat_cols:
        le = LabelEncoder()
        # Объединяем train и test для consistent encoding
        combined = pd.concat([X[col], test_df[col]], axis=0)
        le.fit(combined)
        
        X[col] = le.transform(X[col])
        test_df[col] = le.transform(test_df[col])
        le_dict[col] = le
    
    print("Step 3: Splitting data...")
    # Разделяем на train/validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("Step 4: Training model...")
    # Обучение LightGBM с проверкой версии
    try:
        # Пробуем новый API
        model = lgb.LGBMClassifier(
            n_estimators=1000,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        
        # Пробуем разные варианты параметра early stopping
        try:
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric='auc',
                early_stopping_rounds=50,
                verbose=50
            )
        except TypeError:
            # Если не работает early_stopping_rounds, пробуем early_stopping
            try:
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    eval_metric='auc',
                    early_stopping=50,
                    verbose=50
                )
            except TypeError:
                # Если ничего не работает, обучаем без early stopping
                print("Early stopping not supported, training without it...")
                model.fit(X_train, y_train, verbose=50)
                
    except Exception as e:
        print(f"Error with LGBMClassifier: {e}")
        print("Trying alternative approach...")
        
        # Альтернативный подход с DecisionTree
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
    
    print("Step 5: Evaluation...")
    # Оценка модели
    val_pred = model.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, val_pred)
    print(f'Validation AUC: {val_auc:.5f}')
    
    print("Step 6: Making predictions...")
    # Предсказание на тесте
    test_pred = model.predict_proba(test_df)[:, 1]
    
    # Создание сабмита
    submission = pd.read_csv('src/ctr_sample_submission.csv')
    submission['click'] = test_pred
    submission.to_csv('my_submission.csv', index=False)
    print("Submission file created!")
    print(f"Final AUC: {val_auc:.5f}")

if __name__ == "__main__":
    main()