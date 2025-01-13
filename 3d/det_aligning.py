import cv2
import numpy as np
# import matplotlib.pyplot as plt

def preprocess_image(image_path):
    """
    Предобработка изображения: нахождение угла поворота для выравнивания объекта по вертикали.
    """
    # Чтение изображения
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Изображение по пути {image_path} не найдено")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Бинаризация для выделения объекта
    _, binary = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)

    # Нахождение контуров
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Фильтрация контуров по площади
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 500]
    if not contours:
        raise ValueError("Объект не найден на изображении")

    # Берем самый большой контур
    cnt = max(contours, key=cv2.contourArea)

    # Минимальный ограничивающий прямоугольник
    rect = cv2.minAreaRect(cnt)
    angle = rect[2]  # Угол прямоугольника

    return image, angle, rect

def rotate_image(image, angle):
    """
    Поворот изображения с добавлением отступов, чтобы избежать обрезки.
    """
    # Корректировка угла поворота к ближайшей оси
    if angle>45:
        angle -= 90
    
    h, w = image.shape[:2]
    diagonal = int(np.sqrt(h**2 + w**2))
    padded_image = cv2.copyMakeBorder(
        image,
        top=(diagonal - h) // 2,
        bottom=(diagonal - h) // 2,
        left=(diagonal - w) // 2,
        right=(diagonal - w) // 2,
        borderType=cv2.BORDER_CONSTANT,
        value=(0, 0, 0)
    )
    center = (padded_image.shape[1] // 2, padded_image.shape[0] // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(padded_image, M, (padded_image.shape[1], padded_image.shape[0]), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    gray_rotated = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
    _, binary_rotated = cv2.threshold(gray_rotated, 30, 255, cv2.THRESH_BINARY)
    x, y, w, h = cv2.boundingRect(binary_rotated)
    return rotated[y:y+h, x:x+w]

# def angle_size_check(image):

#     """
#     Вывод угла наклона минимального ограничивающего прямоугольника, его ширины и высоты.
#     """

#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     # Бинаризация для выделения объекта
#     _, binary = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)

#     # Нахождение контуров
#     contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
#     # Фильтрация контуров по площади
#     contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 500]
#     if not contours:
#         raise ValueError("Объект не найден на изображении")

#     # Берем самый большой контур
#     cnt = max(contours, key=cv2.contourArea)

#     # Минимальный ограничивающий прямоугольник
#     rect = cv2.minAreaRect(cnt)
#     angle = rect[2]  # Угол прямоугольника
#     width, height = rect[1]

#     print(f"Угол поворота: {angle}")
#     print(f"Ширина: {width}, Высота: {height}")

# Пути к изображениям
# image1_path = "1.jpg"
# image1_path = "4.png"

def align(img):
    # Предобработка изображений
    image1, angle1, rect1 = preprocess_image(img)

    # angle_size_check(image1)

    # Поворот изображений
    image1 = rotate_image(image1, angle1)
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    return image1

# # Отображение результатов
# plt.figure(figsize=(10, 5))
# plt.title("Result")
# plt.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
# plt.axis("off")
# plt.show()


