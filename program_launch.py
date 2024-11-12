import sys
import os
from predict_detail_type import DetailClassifier
from defected_detail_check import FeatureMatcher

# Добавление директории для импорта класса предобработки изображений (удаление фона)
sys.path.append(os.path.abspath('./preprocess'))
from preprocess.back_delete import ImageProcessor

# Пути к сохранённой модели и LabelEncoder
model_path = './models/random_forest_model.pkl'
le_path = './models/label_encoder.pkl'
img_path = r"./17.jpg"

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

'''сравнение с эталоном'''
if label is not None:
    print(f"Класс: {label}, Уверенность: {confidence}")
    matcher = FeatureMatcher()
    result = matcher.evaluate_image(label, result)
    print(result)

