import sys
import os

from back_delete import ImageProcessor
from predict_detail_type import DetailClassifier
# Добавление директории для импорта класса экстрактора признаков
sys.path.append(os.path.abspath('./descriptors'))
from defected_detail_check import FeatureMatcher


# Пути к сохранённой модели и LabelEncoder
model_path = 'random_forest_model.pkl'
le_path = 'label_encoder.pkl'
img_path = r"D:\Desktop\ref\details\13.jpg"

if not os.path.exists(img_path):
    print("Ошибка: файл не найден.")
    sys.exit(0)  # Завершение программы с кодом ошибки 1

'''удаляем фон'''
processor = ImageProcessor(img_path)
results = processor.process_image()
result = results["result"]

'''классифицируем деталь'''
classifier = DetailClassifier(model_path, le_path)
label, confidence = classifier.classify_image(result)
print(f"Класс: {label}, Уверенность: {confidence}")

'''сравнение с эталоном'''
matcher = FeatureMatcher()
result = matcher.evaluate_image(label, result)
print(result)
