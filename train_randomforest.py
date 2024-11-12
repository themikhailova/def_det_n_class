import sys
import os
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split, LeaveOneOut
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import pickle
from preprocess_image import ImagePreprocessor

# Подключение директорий для импорта классов для предобработки изображений, экстракции дескрипторов и аугментации
sys.path.append(os.path.abspath('./preprocess'))
sys.path.append(os.path.abspath('./descriptors'))
sys.path.append(os.path.abspath('./augmentation'))
from preprocess.preprocess_image import ImagePreprocessor
from descriptors.image_descriptor_extraction import ImageDescriptorExtractor
from descriptors.manage_standard_descriptors import DescriptorManager
from augmentation.image_augmentor import ImageAugmentor

class RandomForestTrainer:
    def __init__(self):
        """
        Инициализация класса для тренировки модели случайного леса.
        """
        self.dataset_path = './processed_details'  # Путь к данным для обучения
        self.features = []
        self.labels = []
        self.label_encoder = LabelEncoder()
        self.preprocessor = ImagePreprocessor()
        self.augmentor = ImageAugmentor()
        self.descriptor_extractor = ImageDescriptorExtractor()
        self.standard_manager = DescriptorManager()  # Загрузка эталонных дескрипторов

    def average_descriptors(self, descriptors):
        """
        Усреднение дескрипторов, если их несколько для одного изображения.
        Если дескрипторов нет, возвращается вектор с нулями фиксированного размера.

        :param descriptors: набор дескрипторов для изображения.
        :return: усредненный дескриптор.

        Усреднение необходимо для того, чтобы преобразовать различные дескрипторы с одного изображения в единую фиксированную форму. 
        Это позволяет гарантировать, что количество признаков всегда одинаково, что критично для обучения моделей машинного обучения.

        Для классификации усреднение помогает создать стабильный и универсальный вектор признаков для каждого изображения, 
        что повышает эффективность обучения. Оно не влияет напрямую на точность, но улучшает общее качество обучения, 
        делая признаки более устойчивыми и упрощая обработку изображений с различным количеством дескрипторов.
        """
        if descriptors is not None and len(descriptors) > 0:
            descriptors = np.array(descriptors)
            return np.mean(descriptors, axis=0)
        else:
            return np.zeros(128)  # примерный размер для SIFT

    def load_and_process_images(self):
        """
        Загрузка и обработка эталонных изображений, включая аугментацию и извлечение дескрипторов.
        """
        # Загрузка эталонных дескрипторов из директории
        ref_features, ref_labels = self.standard_manager.load_standard_descriptors()
        
        # Обработка эталонных дескрипторов 
        for descriptors in ref_features:
            averaged_descriptors = self.average_descriptors(descriptors)  # Усреднение дескрипторов
            self.features.append(averaged_descriptors)
        
        # Применяем метки для эталонных данных
        self.labels.extend(ref_labels)

        # Загрузка и обработка изображений с аугментацией
        for label_folder in os.listdir(self.dataset_path):
            label_path = os.path.join(self.dataset_path, label_folder)
            if os.path.isdir(label_path):
                for filename in os.listdir(label_path):
                    file_path = os.path.join(label_path, filename)
                    image = cv2.imread(file_path)
                    if image is not None:
                        processed_image = self.preprocessor.preprocess_image(image)  # Центрирование и выравнивание объекта на изображении
                        feature_vector = self.descriptor_extractor.extract_descriptors(processed_image)
                        averaged_feature_vector = self.average_descriptors(feature_vector)  # Усреднение
                        self.features.append(averaged_feature_vector)
                        self.labels.append(label_folder)
                        
                        # Аугментация изображения
                        augmented_images = self.augmentor.augment_image(processed_image, filename)
                        for aug_img in augmented_images:
                            aug_feature_vector = self.descriptor_extractor.extract_descriptors(aug_img)
                            averaged_aug_feature_vector = self.average_descriptors(aug_feature_vector)  # Усреднение
                            self.features.append(averaged_aug_feature_vector)
                            self.labels.append(label_folder)

    def train_model(self):
        """
        Обучение модели случайного леса и сохранение модели.
        """
        self.labels = self.label_encoder.fit_transform(self.labels)
        X_train, X_test, y_train, y_test = train_test_split(self.features, self.labels, test_size=0.2, random_state=42)
        
        # Параметры для настройки модели случайного леса с помощью GridSearchCV
        param_grid = {
            'n_estimators': [50, 100, 200],  # Количество деревьев в лесу. Чем больше, тем стабильнее модель, но медленнее обучение.
            'max_depth': [None, 10, 20, 30],  # Максимальная глубина дерева. Ограничивает количество уровней в дереве. 'None' означает отсутствие ограничений.
            'min_samples_split': [2, 3, 5, 10],  # Минимальное количество объектов для разделения узла. Чем больше это значение, тем "сдержаннее" будут деревья.
            'min_samples_leaf': [1, 2, 4],  # Минимальное количество объектов в листьях дерева. Этот параметр помогает избежать переобучения.
            'max_features': ['sqrt', 'log2']  # Количество признаков, которые могут быть использованы для разделения в каждом узле. 'sqrt' и 'log2' - это частые значения для случайного леса.
        }
        
        # Обучение модели с использованием GridSearchCV и LeaveOneOut
        clf = RandomForestClassifier(random_state=42)
        loo = LeaveOneOut()  # Кросс-валидация "Оставить один", проверяет модель на каждом примере по очереди.
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

        with open('./models/random_forest_model.pkl', 'wb') as model_file:
            pickle.dump(best_model, model_file)
        with open('./models/label_encoder.pkl', 'wb') as le_file:
            pickle.dump(self.label_encoder, le_file)


if __name__ == "__main__":
    # Инициализация класса обучения
    trainer = RandomForestTrainer()
    
    # Загрузка и обработка изображений
    trainer.load_and_process_images()

    # Обучение модели
    trainer.train_model()
