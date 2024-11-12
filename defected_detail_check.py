import os
import cv2
import numpy as np
import sys
from preprocess_image import ImagePreprocessor
from predict_detail_type import DetailClassifier

# Добавление директории для импорта класса экстрактора признаков
sys.path.append(os.path.abspath('./descriptors'))
from descriptors.image_descriptor_extraction import ImageDescriptorExtractor
from descriptors.manage_standard_descriptors import DescriptorManager

class FeatureMatcher:
    def __init__(self, good_match_threshold=0.75, match_ratio_threshold=0.7):
        """
        Инициализация FeatureMatcher. Для опрделения деффектности модели

        :param good_match_threshold: порог для фильтрации хороших совпадений методом Lowe's Ratio Test
        :param match_ratio_threshold: порог для определения качества детали
        """
        self.good_match_threshold = good_match_threshold
        self.match_ratio_threshold = match_ratio_threshold
        self.detector = ImageDescriptorExtractor()  
        self.matcher = self._initialize_matcher()
        self.preprocessor = ImagePreprocessor()
        self.reference_manager = DescriptorManager()  

    def _initialize_matcher(self):
        """
        Инициализация FLANN-based matcher.

        :return: FLANN-based matcher
        """
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        return cv2.FlannBasedMatcher(index_params, search_params)

    def load_class_desc(self, class_number):
        """
        Загрузка эталонных дескрипторов указанного класса.

        :param class_number: номер класса для загрузки дескрипторов
        :return: список дескрипторов всех изображений данного класса
        """
        features, labels = self.reference_manager.load_reference_descriptors()
        class_desc = [desc for desc, label in zip(features, labels) if label == class_number]
        return class_desc

    def get_image_desc(self, image):
        """
        Получение дескрипторов нового изображения.

        :param image: обработанное изображение
        :return: дескрипторы
        """
        return self.detector.extract_descriptors(image)

    def is_perfect_match(self, new_desc, class_desc):
        """
        Оценка совпадений для определения качества детали.
        Усреднение дескрипторов перед сравнением здесь делать не надо, тут это снизит точность

        :param new_desc: дескрипторы нового изображения
        :param class_desc: список дескрипторов эталонного класса
        :return: True, если деталь идеальная; False, если нет
        """
        if len(new_desc) == 0:
            return False  # Если нет дескрипторов, сразу считаем, что совпадений нет

        good_matches_total = 0

        for ref_desc in class_desc:
            matches = self.matcher.knnMatch(new_desc, ref_desc, k=2)
            good_matches = [m for m, n in matches if m.distance < self.good_match_threshold * n.distance]
            good_matches_total += len(good_matches)

        match_ratio = good_matches_total / len(new_desc) if len(new_desc) > 0 else 0
        return match_ratio > self.match_ratio_threshold

    def evaluate_image(self, class_number, image):
        """
        Оценка нового изображения для проверки наличия дефектов на основе классификации.

        :param class_number: номер класса для сравнения
        :param image: изображение для оценки
        :return: результат проверки ("идеальная" или "с дефектами")
        """
        class_desc = self.load_class_desc(class_number)
        new_desc = self.get_image_desc(image)

        if self.is_perfect_match(new_desc, class_desc):
            return "Деталь идеальная"
        else:
            return "Деталь имеет дефекты"


# if __name__ == "__main__":
#     # Пути к сохранённой модели и LabelEncoder
#     model_path = 'random_forest_model.pkl'
#     le_path = 'label_encoder.pkl'

#     # Инициализация классификатора
#     classifier = DetailClassifier(model_path, le_path)

#     # Пути к изображению для классификации
#     image_path = '.\\test_images_unlabeled\\17.jpg'
#     test_image = cv2.imread(image_path)

#     # Классификация изображения с использованием DetailClassifier
#     class_number, confidence = classifier.classify_image(test_image)

#     # Проверка на null классов
#     if class_number is not None:
#         # Инициализация FeatureMatcher
#         matcher = FeatureMatcher()

#         # Оценка изображения с использованием класса и пути
#         result = matcher.evaluate_image(class_number, test_image)
#         print(result)