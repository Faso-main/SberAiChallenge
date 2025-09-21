import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import gc  # Для очистки памяти

# Загружаем данные (это займет время и память)
# Попробуйте сначала на sample данных (например, на 1 млн строк), чтобы отладить пайплайн.
dtypes = {
    'id': 'str',
    'click': 'uint8',
    'hour': 'uint32',
    'C1': 'uint32',
    'banner_pos': 'uint32',
    'site_id': 'str',
    'site_domain': 'str',
    'site_category': 'str',
    'app_id': 'str',
    'app_domain': 'str',
    'app_category': 'str',
    'device_id': 'str',
    'device_ip': 'str',
    'device_model': 'str',
    'device_type': 'uint32',
    'device_conn_type': 'uint32',
    'C14': 'uint32',
    'C15': 'uint32',
    'C16': 'uint32',
    'C17': 'uint32',
    'C18': 'uint32',
    'C19': 'uint32',
    'C20': 'uint32',
    'C21': 'uint32',
}

print("Loading data...")
train_df = pd.read_csv('src/ctr_train.csv', dtype=dtypes, nrows=40000000) # Используйте nrows для теста
test_df = pd.read_csv('src/ctr_test.csv', dtype=dtypes)

# Предобработка признака 'hour'
def process_hour(df):
    df['hour'] = df['hour'].astype(str)
    df['hour_weekday'] = pd.to_datetime(df['hour'].str[:6], format='%y%m%d').dt.weekday
    df['hour_hour'] = pd.to_numeric(df['hour'].str[6:])
    return df.drop(columns=['hour'])

train_df = process_hour(train_df)
test_df = process_hour(test_df)

# Удаляем колонки, которые скорее всего являются уникальными ID
# Эти признаки имеют слишком высокую кардинальность и приведут к переобучению.
cols_to_drop = ['id', 'device_id', 'device_ip']
train_df.drop(columns=cols_to_drop, inplace=True)
test_df.drop(columns=cols_to_drop, inplace=True)

# Определяем категориальные и числовые признаки
cat_cols = [col for col in train_df.columns if (train_df[col].dtype == 'object' or col in ['C1', 'banner_pos', 'device_type', 'device_conn_type']) and col != 'click']
num_cols = [col for col in train_df.columns if train_df[col].dtype in ['int64', 'uint32', 'uint8'] and col not in cat_cols and col != 'click']
target = 'click'

print("Categorical columns:", cat_cols)
print("Numerical columns:", num_cols)

# Функция для Mean Encoding с регуляризацией (сглаживание)
def mean_encoding_smooth(train_df, test_df, col, target, alpha=100):
    # Вычисляем глобальное среднее
    global_mean = train_df[target].mean()
    # Вычисляем среднее и количество для каждой категории в тренировочных данных
    agg = train_df.groupby(col)[target].agg(['mean', 'count'])
    # Вычисляем сглаженное среднее
    agg['smooth_mean'] = (agg['count'] * agg['mean'] + alpha * global_mean) / (agg['count'] + alpha)
    smooth_mean = agg['smooth_mean'].to_dict()
    # Применяем к train и test
    train_df[col + '_mean'] = train_df[col].map(smooth_mean)
    test_df[col + '_mean'] = test_df[col].map(smooth_mean)
    # Заполняем пропуски в test глобальным средним
    test_df[col + '_mean'].fillna(global_mean, inplace=True)
    return train_df, test_df

# Применяем Mean Encoding ко всем категориальным признакам
print("Starting Mean Encoding...")
for col in cat_cols:
    print(f"Encoding {col}...")
    train_df, test_df = mean_encoding_smooth(train_df, test_df, col, target)
    # Удаляем исходные категориальные колонки
    train_df.drop(columns=[col], inplace=True)
    test_df.drop(columns=[col], inplace=True)

# Теперь все признаки - числа
X = train_df.drop(columns=[target])
y = train_df[target]
X_test = test_df

# Освобождаем память
del train_df, test_df
gc.collect()

# Разбиваем данные на тренировочную и валидационную части для оценки
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Создаем датасеты для LightGBM
lgb_train = lgb.Dataset(X_train, y_train, free_raw_data=False)
lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train, free_raw_data=False)

# Параметры модели LightGBM
params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'num_leaves': 63,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'seed': 42,
    'verbose': -1,
    'n_jobs': -1,
}

print("Training LightGBM model...")
# Обучаем модель с ранней остановкой
model = lgb.train(
    params,
    lgb_train,
    num_boost_round=10000,
    valid_sets=[lgb_train, lgb_val],
    valid_names=['train', 'valid'],
    early_stopping_rounds=50,
    verbose_eval=100
)

# Предсказываем на валидации для проверки AUC
val_pred = model.predict(X_val, num_iteration=model.best_iteration)
val_auc = roc_auc_score(y_val, val_pred)
print(f'Validation AUC: {val_auc:.5f}')

# Предсказываем на тестовом наборе
print("Making predictions on test set...")
test_pred = model.predict(X_test, num_iteration=model.best_iteration)

# Готовим сабмит
submission = pd.read_csv('src/ctr_sample_submission.csv')
submission['click'] = test_pred
submission.to_csv('src/my_submission.csv', index=False)
print("Submission file 'my_submission.csv' is ready!")