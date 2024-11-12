import cv2
import numpy as np

class ImagePreprocessor:
    @staticmethod
    def preprocess_image(image):
        """
        Центрирование и выравнивание объекта на изображении.

        :param image: входное изображение
        :return: предварительно обработанное изображение
        """
        # Преобразование изображения в оттенки серого
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Бинаризация изображения для выделения контура
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Поиск контуров
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Находим самый крупный контур
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Центрирование изображения
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:  # Проверка на нулевой момент
                cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                h, w = gray.shape
                shift_x, shift_y = w // 2 - cx, h // 2 - cy
                # Центрирование
                translation_matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
                gray = cv2.warpAffine(gray, translation_matrix, (w, h))
            
            # Вычисление угла наклона и выравнивание
            angle = cv2.minAreaRect(largest_contour)[2]
            angle = angle + 90 if angle < -45 else angle  # Обеспечение корректности угла
            rotation_matrix = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            gray = cv2.warpAffine(gray, rotation_matrix, (w, h))
        
        return gray

# if __name__ == "__main__":
#     # Пример использования
#     image = cv2.imread('path_to_image.jpg')  # Замените на путь к вашему изображению
#     preprocessor = ImagePreprocessor()
#     processed_image = preprocessor.preprocess_image(image)
    
#     # Сохранение или отображение результата
#     cv2.imwrite('processed_image.jpg', processed_image)  # Сохранение обработанного изображения
#     cv2.imshow('Processed Image', processed_image)  # Отображение обработанного изображения
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
