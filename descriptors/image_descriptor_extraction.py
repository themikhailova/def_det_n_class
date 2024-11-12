import cv2
import numpy as np

class ImageDescriptorExtractor:
    def __init__(self, algorithm='SIFT'):
        """
        Инициализирует экстрактор дескрипторов на основе указанного алгоритма.
        
        :param algorithm: Алгоритм извлечения дескрипторов, может быть 'SIFT' или 'ORB'
        """
        
        # Выбор алгоритма извлечения на основе параметра 'algorithm'
        if algorithm == 'SIFT':
            self.extractor = cv2.SIFT_create()
        elif algorithm == 'ORB':
            self.extractor = cv2.ORB_create()
        else:
            raise ValueError("Неопределенный алгоритм. Используйте 'SIFT' или 'ORB'.")

    def extract_descriptors(self, image):
        """
        Извлечение дескрипторов из изображения.
        
        :param image: изображение в формате numpy.ndarray
        :return: дескрипторы изображения или пустой массив, если дескрипторы не найдены
        """
        # Находим ключевые точки и вычисляем дескрипторы
        keypoints, descriptors = self.extractor.detectAndCompute(image, None)
        return descriptors if descriptors is not None else np.array([])
