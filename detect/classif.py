import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV, LeaveOneOut
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import numpy as np

def train_model(file_path):
    """
    Обучение модели случайного леса на данных из CSV-файла и сохранение модели.
    """
    # Загрузка данных из файла
    data = pd.read_excel(file_path)
    
    # Разделение данных на признаки и метки
    features = data.iloc[:, :-1]  # Все колонки, кроме последней
    print(features)
    labels = data.iloc[:, -1]  # Последняя колонка

    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)

    # Разделение на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    print(X_train)
    # print(y_train.shape)    
    # Параметры для настройки модели случайного леса с помощью GridSearchCV
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 3, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }
    
    # Обучение модели с использованием GridSearchCV и LeaveOneOut
    clf = RandomForestClassifier(random_state=42)
    loo = LeaveOneOut()
    grid_search = GridSearchCV(clf, param_grid, cv=loo, scoring='accuracy', n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    
    # Оценка точности и сохранение модели
    best_model = grid_search.best_estimator_
    accuracy = accuracy_score(y_test, best_model.predict(X_test))
    print(f"Точность на тестовых данных: {accuracy * 100:.2f}%")

    # Вывод лучших подобранных параметров
    print("Лучшие параметры модели:")
    for param, value in grid_search.best_params_.items():
        print(f"{param}: {value}")

    # Сохранение модели и энкодера
    with open('./models/random_forest_model.pkl', 'wb') as model_file:
        pickle.dump(best_model, model_file)
    with open('./models/label_encoder.pkl', 'wb') as le_file:
        pickle.dump(label_encoder, le_file)

# Пример вызова функции
# train_model('./anomalies.xlsx')  # Укажите путь к вашему файлу CSV

def classify_image(features, threshold=0.6):
    """
        Классификация изображения с учётом порога уверенности.

        :param image: Изображение для классификации
        :return: Кортеж (предсказанный класс, уверенность), или (None, уверенность) при низкой уверенности
    """
    with open('./models/random_forest_model.pkl', 'rb') as model_file:
            loaded_model = pickle.load(model_file)
        
    with open('./models/label_encoder.pkl', 'rb') as le_file:
        loaded_label_encoder = pickle.load(le_file)

    # Предсказания и вероятности с использованием модели
    probs = loaded_model.predict_proba(features)
    max_prob = np.max(probs)  # Максимальная вероятность
    predicted_label = loaded_model.predict(features)[0]  # Класс с максимальной вероятностью
    predicted_label = loaded_label_encoder.inverse_transform([predicted_label])[0]

    # Если уверенность выше порога, возвращаем результат
    if max_prob >= threshold:
            #print(f"Предсказанный класс: {predicted_label}, Уверенность: {max_prob:.2f}")
        return predicted_label, max_prob  # Возвращаем числовую метку класса
    else:
        # Если уверенность ниже порога, классификация не удалась
        print("Не удалось классифицировать изображение.")
    return None, max_prob


# Подготовка новых данных
new_data = [[9.5878794,	0.754464765, 0.571428571, 216.452381]]  # Пример данных одной фотографии
df_new_data = pd.DataFrame(new_data, columns=['Compactness', 'Eccentricity', 'Aspect Ratio', 'Mean Intensity'])  # Названия должны совпадать с обучающими
print(classify_image(df_new_data))

# # Загрузка модели и энкодера
# with open('./models/random_forest_model.pkl', 'rb') as model_file:
#     model = pickle.load(model_file)
# with open('./models/label_encoder.pkl', 'rb') as le_file:
#     label_encoder = pickle.load(le_file)

# # Подготовка новых данных
# new_data = [[9.5878794,	0.754464765, 0.571428571, 216.452381]] # Пример данных одной фотографии
# df_new_data = pd.DataFrame(new_data, columns=['Compactness', 'Eccentricity', 'Aspect Ratio', 'Mean Intensity'])  # Названия должны совпадать с обучающими
# print(df_new_data)
# # Прогнозирование
# predictions = model.predict(df_new_data)

# # Если нужно декодировать предсказания
# decoded_predictions = label_encoder.inverse_transform(predictions)
# print(decoded_predictions)




'''
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

def load_data(file_path):
    data = pd.read_excel(file_path)
    features = data[['Compactness', 'Eccentricity', 'Aspect Ratio', 'Mean Intensity']]
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

def save_models(base_models, meta_model, label_encoder, save_path='./models'):
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

file_path = './anomalies.xlsx'
main(file_path)

'''
