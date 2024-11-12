import cv2
import numpy as np

class ImageDescriptorExtractor:
    def __init__(self, algorithm='SIFT'):
        """
        Инициализирует извлечение дескрипторов на основе указанного алгоритма.
        
        :param algorithm: алгоритм извлечения дескрипторов, может быть 'SIFT' или 'ORB'
        
        SIFT (Scale-Invariant Feature Transform): Метод, который находит ключевые точки и генерирует дескрипторы, 
        инвариантные к масштабу и повороту. Обычно подходит для задач, где важно распознавание объекта при изменении 
        масштаба, освещения и угла зрения.
        
        ORB (Oriented FAST and Rotated BRIEF): Более быстрый и легковесный алгоритм, который также устойчив к поворотам, 
        но менее чувствителен к изменениям масштаба и освещения по сравнению с SIFT. 
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
        
        Процесс включает два этапа: обнаружение ключевых точек на изображении и вычисление дескрипторов, 
        которые могут быть использованы для сравнения и сопоставления объектов на других изображениях.
        """
        # Находим ключевые точки и вычисляем дескрипторы
        keypoints, descriptors = self.extractor.detectAndCompute(image, None)
        return descriptors if descriptors is not None else np.array([])
