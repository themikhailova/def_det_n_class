import cv2
import numpy as np
import pickle
import sys
import os
from preprocess_image import ImagePreprocessor

# Добавление директории для импорта класса экстрактора признаков
sys.path.append(os.path.abspath('./descriptors'))
from descriptors.image_descriptor_extraction import ImageDescriptorExtractor


class DetailClassifier:
    def __init__(self, model_path, le_path, threshold=0.6):
        """
        Инициализация классификатора с загрузкой модели и LabelEncoder.

        :param model_path: Путь к сохранённой модели
        :param le_path: Путь к сохранённому LabelEncoder
        :param threshold: Порог уверенности для классификации (по умолчанию 0.6)
        """
        # Загрузка модели и LabelEncoder
        with open(model_path, 'rb') as model_file:
            self.loaded_model = pickle.load(model_file)
        
        with open(le_path, 'rb') as le_file:
            self.loaded_label_encoder = pickle.load(le_file)
        
        # Инициализация экстрактора признаков
        self.descriptor_extractor = ImageDescriptorExtractor()  # По умолчанию используется SIFT или ORB
        self.threshold = threshold  # Порог уверенности для классификации

    def preprocess_image(self, image):
        """
        Предобработка изображения с помощью ImagePreprocessor.

        :param image: Входное изображение
        :return: Обработанное изображение
        """
        # Предобработка с центровкой и выравниванием объекта
        return ImagePreprocessor.preprocess_image(image)
    
    def extract_features(self, image):
        """
        Извлечение признаков из изображения с помощью ImageDescriptorExtractor.

        :param image: Входное изображение
        :return: Усреднённый вектор признаков изображения
        """
        descriptors = self.descriptor_extractor.extract_descriptors(image)
        
        # Если дескрипторы найдены, возвращаем их усреднённый вектор
        if descriptors.size > 0:
            return np.mean(descriptors, axis=0)  # Усредняем по всем дескрипторам
        # Если дескрипторы не найдены, возвращаем пустой вектор
        return np.zeros((1,))  # Пустой вектор, если дескрипторов нет

    def classify_image(self, image):
        """
        Классификация изображения с учётом порога уверенности.

        :param image: Изображение для классификации
        :return: Кортеж (предсказанный класс, уверенность), или (None, уверенность) при низкой уверенности
        """
        preprocessed_image = self.preprocess_image(image)
        feature_vector = self.extract_features(preprocessed_image).reshape(1, -1)
        
        # Если извлечены признаки, продолжаем классификацию
        if feature_vector.size == 0:
            return None, 0.0  # Возвращаем None, если дескрипторов нет

        # Предсказания и вероятности с использованием модели
        probs = self.loaded_model.predict_proba(feature_vector)
        max_prob = np.max(probs)  # Максимальная вероятность
        predicted_label = self.loaded_model.predict(feature_vector)[0]  # Класс с максимальной вероятностью
        predicted_label = self.loaded_label_encoder.inverse_transform([predicted_label])[0]

        # Если уверенность выше порога, возвращаем результат
        if max_prob >= self.threshold:
            #print(f"Предсказанный класс: {predicted_label}, Уверенность: {max_prob:.2f}")
            return predicted_label, max_prob  # Возвращаем числовую метку класса
        else:
            # Если уверенность ниже порога, классификация не удалась
            print("Не удалось классифицировать изображение.")
            return None, max_prob


# Пример использования:
# if __name__ == "__main__":
#     # Пути к сохранённой модели и LabelEncoder
#     model_path = './models/random_forest_model.pkl'
#     le_path = './models/label_encoder.pkl'
    
#     # Инициализация классификатора без необходимости передавать descriptor_extractor
#     classifier = DetailClassifier(model_path, le_path)
    
#     test_image = cv2.imread('.\\test_images_unlabeled\\17.jpg')
#     label, confidence = classifier.classify_image(test_image)
#     print(f"Класс: {label}, Уверенность: {confidence}")
