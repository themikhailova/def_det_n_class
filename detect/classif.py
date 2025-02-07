import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import os


import pickle
import numpy as np
import cv2

# Загрузка моделей
def load_trained_models(model_path='./detect/models'):
    models = {}
    for model_name in ['rf_model.pkl', 'gb_model.pkl', 'knn_model.pkl', 'meta_model.pkl', 'label_encoder.pkl']:
        with open(f"{model_path}/{model_name}", 'rb') as file:
            models[model_name] = pickle.load(file)
    return models

FEATURE_COLUMNS = [
    'area', 'perimeter', 'relative_area', 'relative_perimeter', 'relative_centroid_distance', 'compactness', 'aspect_ratio', 'eccentricity', 'equivalent_diameter', 'angularity', 'complexity', 'fourier_3', 'fourier_5', 'fourier_7', 'fourier_9', 'mean_intensity', 'median_intensity', 'std_intensity', 'max_intensity', 'min_intensity', 'uniformity', 'entropy', 'mean_gradient', 'std_gradient'
]
# Функция классификации аномалий
def classify_anomaly(features):
    models = load_trained_models()
    
    base_models = [models['rf_model.pkl'], models['gb_model.pkl'], models['knn_model.pkl']]
    meta_model = models['meta_model.pkl']
    label_encoder = models['label_encoder.pkl']

    # Создаём DataFrame с правильными именами столбцов
    feature_values = pd.DataFrame([features], columns=FEATURE_COLUMNS)

    # Получаем предсказания базовых моделей
    base_predictions = np.column_stack([model.predict_proba(feature_values)[:, 1] for model in base_models])

    # Делаем финальное предсказание с мета-моделью
    final_prediction = meta_model.predict(base_predictions)
    anomaly_type = label_encoder.inverse_transform(final_prediction)[0]

    # Назначаем цвет для типа аномалии
    anomaly_colors = {
        'leakage': (255, 0, 0),  
        'fistulas': (0, 255, 0),  
        'bevel': (0, 255, 255),  
        'scratch': (255, 165, 0),  
        'no': (0, 0, 255),  
    }
    
    colour = anomaly_colors.get(anomaly_type, (128, 128, 128))  # Серый, если тип неизвестен

    return anomaly_type, colour

def load_data(file_path):
    data = pd.read_excel(file_path, engine="openpyxl")
    features = data[['area', 'perimeter', 'relative_area', 'relative_perimeter', 'relative_centroid_distance', 'compactness', 'aspect_ratio', 'eccentricity', 'equivalent_diameter', 'angularity', 'complexity', 'fourier_3', 'fourier_5', 'fourier_7', 'fourier_9', 'mean_intensity', 'median_intensity', 'std_intensity', 'max_intensity', 'min_intensity', 'uniformity', 'entropy', 'mean_gradient', 'std_gradient']]
    labels = data['Y']
    return features, labels

def preprocess_data(features, labels):
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42, stratify=labels)
    return X_train, X_test, y_train, y_test, label_encoder

def perform_grid_search(model, param_grid, X_train, y_train):
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    print(f"Лучшие параметры для {model.__class__.__name__}: {grid_search.best_params_}")
    return grid_search.best_estimator_

def define_models_with_tuning(X_train, y_train):
    # Random Forest
    rf_param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    rf = RandomForestClassifier(random_state=42)
    rf_best = perform_grid_search(rf, rf_param_grid, X_train, y_train)
    
    # Gradient Boosting
    gb_param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    }
    gb = GradientBoostingClassifier(random_state=42)
    gb_best = perform_grid_search(gb, gb_param_grid, X_train, y_train)
    
    # knn
    knn_param_grid = {
        'n_neighbors': [3, 5, 10, 15],
        'weights': ['uniform', 'distance'],
        'p': [1, 2]
    }
    knn = KNeighborsClassifier()
    knn_best = perform_grid_search(knn, knn_param_grid, X_train, y_train)
    
    # список базовых моделей
    base_models = [
        ('rf', rf_best),
        ('gb', gb_best),
        ('knn', knn_best)
    ]
    
    # модель-мета
    meta_model = LogisticRegression(random_state=42)
    return base_models, meta_model

def stacking(base_models, meta_model, X_train, y_train, X_test, n_folds=5):
    """
    стэкинг
    """
    train_meta = np.zeros((X_train.shape[0], len(base_models)))
    test_meta = np.zeros((X_test.shape[0], len(base_models)))
    test_meta_single = np.zeros((n_folds, X_test.shape[0], len(base_models)))
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    for i, (name, model) in enumerate(base_models):
        print(f"обучение базовой модели: {name}")
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
            model.fit(X_fold_train, y_fold_train)
            train_meta[val_idx, i] = model.predict_proba(X_fold_val)[:, 1]
            test_meta_single[fold, :, i] = model.predict_proba(X_test)[:, 1]
        test_meta[:, i] = test_meta_single[:, :, i].mean(axis=0)
    
    print("обучение модели-мета...")
    meta_model.fit(train_meta, y_train)
    final_predictions = meta_model.predict(test_meta)
    return final_predictions, base_models, meta_model

def save_models(base_models, meta_model, label_encoder, save_path='./detect/models'):
    os.makedirs(save_path, exist_ok=True)
    
    for name, model in base_models:
        with open(f"{save_path}/{name}_model.pkl", 'wb') as model_file:
            pickle.dump(model, model_file)
    
    with open(f"{save_path}/meta_model.pkl", 'wb') as meta_file:
        pickle.dump(meta_model, meta_file)
    
    with open(f"{save_path}/label_encoder.pkl", 'wb') as encoder_file:
        pickle.dump(label_encoder, encoder_file)
    print("модели сохранены")

def evaluate_model(y_test, predictions):
    accuracy = accuracy_score(y_test, predictions)
    print(f"Точность стэкинг-модели: {accuracy * 100:.2f}%")

def main(file_path):
    features, labels = load_data(file_path)
    X_train, X_test, y_train, y_test, label_encoder = preprocess_data(features, labels)
    base_models, meta_model = define_models_with_tuning(X_train, y_train)
    predictions, trained_base_models, trained_meta_model = stacking(base_models, meta_model, X_train, y_train, X_test)
    evaluate_model(y_test, predictions)
    save_models(trained_base_models, trained_meta_model, label_encoder)

# file_path = './train.xlsx'
# main(file_path)

TRAINING_DATA_PATH = './train.xlsx'

def update_training_data(new_data_path, training_data_path=TRAINING_DATA_PATH):
    """
    Обновляет тренировочный датасет, добавляя новые данные.
    """
    new_data = pd.read_excel(new_data_path, engine="openpyxl")

    if os.path.exists(training_data_path):
        old_data = pd.read_excel(training_data_path, engine="openpyxl")
        combined_data = pd.concat([old_data, new_data], ignore_index=True)
        combined_data.drop_duplicates(inplace=True)
    else:
        combined_data = new_data

    combined_data.to_excel(training_data_path, index=False, engine="openpyxl")
    print(f"Данные обновлены и сохранены в {training_data_path}")

    return combined_data

def retrain_model(new_data_path, model_path='./detect/models'):
    # Обновляем датасет
    updated_data = update_training_data(new_data_path)
    new_features, new_labels = load_data(TRAINING_DATA_PATH)
    models = load_trained_models(model_path)
    
    base_models = {
        'rf': models['rf_model.pkl'],
        'gb': models['gb_model.pkl'],
        'knn': models['knn_model.pkl']
    }
    meta_model = models['meta_model.pkl']
    label_encoder = models['label_encoder.pkl']
    
    # Объединяем старые и новые метки
    all_labels = np.concatenate([models['label_encoder.pkl'].classes_, new_labels])
    
    label_encoder.fit(all_labels)  # Обновляем encoder
    
    # метки в числовые значения
    new_labels_encoded = label_encoder.transform(new_labels)
    
    old_data_path = './train.xlsx'  
    old_features, old_labels = load_data(old_data_path)
    old_labels_encoded = label_encoder.transform(old_labels)
    
    combined_features = pd.concat([old_features, new_features], ignore_index=True)
    combined_labels = np.concatenate([old_labels_encoded, new_labels_encoded])
    
    X_train, X_test, y_train, y_test = train_test_split(
        combined_features, combined_labels, test_size=0.2, random_state=42, stratify=combined_labels
    )
    
    if len(np.unique(y_train)) < 2:
        print("Недостаточно классов для дообучения. Использую старые данные для балансировки.")
        return
    
    for name, model in base_models.items():
        print(f"Дообучение модели {name}...")
        model.fit(X_train, y_train)
    
    train_meta = np.column_stack([model.predict_proba(X_train)[:, 1] for model in base_models.values()])
    test_meta = np.column_stack([model.predict_proba(X_test)[:, 1] for model in base_models.values()])

    # Дообучение мета-модели
    print("Дообучение мета-модели...")
    meta_model.fit(train_meta, y_train)

    # Оценка
    predictions = meta_model.predict(test_meta)
    evaluate_model(y_test, predictions)

    # Сохранение обновленных моделей
    save_models(base_models.items(), meta_model, label_encoder, model_path)
    print("Дообучение завершено!")
