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