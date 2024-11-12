import os
import pickle
import cv2
import sys

# Подключение директории для импорта класса экстрактора
sys.path.append(os.path.abspath('./descriptors'))
from descriptors.image_descriptor_extraction import ImageDescriptorExtractor

class DescriptorManager:
    def __init__(self, descriptor_extractor=None, dataset_path='./processed_details', descriptor_dir=r'D:\Desktop\ref\standard_desc'):
        """
        Инициализация менеджера эталонных дескрипторов.
        
        :param descriptor_extractor: объект класса ImageDescriptorExtractor (по умолчанию создается новый экземпляр)
        :param dataset_path: путь к директории с эталонными изображениями
        :param descriptor_dir: путь к директории для сохранения дескрипторов
        """
        # Создаем новый экземпляр ImageDescriptorExtractor, если descriptor_extractor не передан
        self.descriptor_extractor = descriptor_extractor if descriptor_extractor is not None else ImageDescriptorExtractor()
        self.dataset_path = dataset_path
        self.descriptor_dir = descriptor_dir

    def save_reference_descriptors(self):
        """
        Извлекает и сохраняет дескрипторы из изображений в dataset_path в директорию descriptor_dir.
        """
        os.makedirs(self.descriptor_dir, exist_ok=True)  # Создание директории, если её нет
        descriptors_saved = False  # Флаг для отслеживания сохранения

        # Проход по каждой поддиректории (метке класса) в dataset_path
        for label_folder in os.listdir(self.dataset_path):
            label_path = os.path.join(self.dataset_path, label_folder)
            if not os.path.isdir(label_path):
                continue

            class_dir = os.path.join(self.descriptor_dir, label_folder)
            os.makedirs(class_dir, exist_ok=True)  # Создаем папку для дескрипторов класса

            for filename in os.listdir(label_path):
                file_path = os.path.join(label_path, filename)
                image = cv2.imread(file_path)
                
                if image is not None:
                    descriptors = self.descriptor_extractor.extract_descriptors(image)
                    if descriptors.size > 0:
                        descriptors_saved = True
                        descriptor_path = os.path.join(class_dir, f"{os.path.splitext(filename)[0]}_desc.pkl")
                        with open(descriptor_path, 'wb') as f:
                            pickle.dump(descriptors, f)
        
        # Сообщение о результате сохранения
        if descriptors_saved:
            print("Дескрипторы успешно сохранены.")
        else:
            print("Не удалось найти дескрипторов для сохранения.")

    def load_reference_descriptors(self):
        """
        Загружает дескрипторы из файлов в директории descriptor_dir.
        
        :return: списки дескрипторов и меток (классов)
        """
        features = []  # Для хранения дескрипторов
        labels = []    # Для хранения меток классов
        descriptors_loaded = False  # Флаг для отслеживания успешной загрузки

        # Проход по каждой папке (классу) в директории descriptor_dir
        for label_folder in os.listdir(self.descriptor_dir):
            label_path = os.path.join(self.descriptor_dir, label_folder)
            if not os.path.isdir(label_path):
                continue
            
            for filename in os.listdir(label_path):
                descriptor_path = os.path.join(label_path, filename)
                with open(descriptor_path, 'rb') as f:
                    descriptors = pickle.load(f)
                    features.append(descriptors)
                    labels.append(label_folder)  # Добавляем метку для каждого дескриптора
                    descriptors_loaded = True

        # Сообщение о результате загрузки
        if descriptors_loaded:
            print("Эталонные дескрипторы успешно загружены.")
        else:
            print("Не удалось загрузить эталонные дескрипторы из директории.")

        return features, labels  # Возвращаем дескрипторы и соответствующие им метки


if __name__ == "__main__":
    # Инициализация менеджера эталонных дескрипторов без параметров (по умолчанию)
    reference_manager = DescriptorManager()
    
    # Сохранение эталонных дескрипторов в директорию
    reference_manager.save_reference_descriptors()
    
    # Загрузка эталонных дескрипторов для последующей обработки
    #features, labels = reference_manager.load_reference_descriptors()
