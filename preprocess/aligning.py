import cv2
import numpy as np
import matplotlib.pyplot as plt

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

def adjust_angles(angle1, angle2, rect1, rect2):
    """
    Корректирует углы поворота изображений на основе их размеров.
    """
    width1, height1 = rect1[1]
    width2, height2 = rect2[1]

    if angle1 < 45 and ((width1 > height1 and width2 < height2) or (width1 < height1 and width2 > height2)):
        angle2 = -90 + angle2

    if angle1 > 45 and ((width1 > height1 and width2 > height2) or (width1 < height1 and width2 < height2)):
        angle1 = -90 + angle1
        angle2 = -90 + angle2

    if angle1 > 45 and ((width1 > height1 and width2 < height2) or (width1 < height1 and width2 > height2)):
        angle1 = -90 + angle1

    return angle1, angle2

def rotate_image(image, angle):
    """
    Поворот изображения с добавлением отступов, чтобы избежать обрезки.
    """
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

def scale_images(image1, image2):
    """
    Приведение двух изображений к одному масштабу на основе площади.
    """
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]
    scale = np.sqrt((h1 * w1) / (h2 * w2))
    new_h2 = int(h2 * scale)
    new_w2 = int(w2 * scale)
    resized_image2 = cv2.resize(image2, (new_w2, new_h2), interpolation=cv2.INTER_CUBIC)
    return image1, resized_image2

def one_direction(image1, image2):
    """
    Унификация направления второй детали относительно первой.
    """
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    _, binary1 = cv2.threshold(gray1, 30, 255, cv2.THRESH_BINARY)
    _, binary2 = cv2.threshold(gray2, 30, 255, cv2.THRESH_BINARY)
    moments1 = cv2.moments(binary1)
    moments2 = cv2.moments(binary2)
    x1_center = moments1["m10"] / moments1["m00"]
    y1_center = moments1["m01"] / moments1["m00"]
    x2_center = moments2["m10"] / moments2["m00"]
    y2_center = moments2["m01"] / moments2["m00"]

    if (y1_center > image1.shape[0] / 2 and y2_center < image2.shape[0] / 2) or \
       (y1_center < image1.shape[0] / 2 and y2_center > image2.shape[0] / 2):
        image2 = cv2.flip(image2, 0)
    if (x1_center > image1.shape[1] / 2 and x2_center < image2.shape[1] / 2) or \
       (x1_center < image1.shape[1] / 2 and x2_center > image2.shape[1] / 2):
        image2 = cv2.flip(image2, 1)

    return image2

# def angle_size_check(image):
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
   
#     return image

# Пути к изображениям
# # image1_path = "4_noback_rot.png"
# image2_path = "4_noback_vert3.png"
# image1_path = "4_noback_rot.png"#"4_noback.png"
image1_path = "green1.jpg"  # Укажите путь к первому изображению
image2_path = "green2.jpg"  # Укажите путь ко второму изображению
# image1_path = "2_noback.jpg"
# image2_path = "3_noback.jpg"

# Предобработка изображений
image1, angle1, rect1 = preprocess_image(image1_path)
image2, angle2, rect2 = preprocess_image(image2_path)

# Корректировка углов
angle1, angle2 = adjust_angles(angle1, angle2, rect1, rect2)

# Поворот изображений
image1 = rotate_image(image1, angle1)
image2 = rotate_image(image2, angle2)

# Унификация направления
image2 = one_direction(image1, image2)

# Приведение к одному масштабу
image1, image2 = scale_images(image1, image2)

# Проверка результатов
# image1 = angle_size_check(image1)
# image2 = angle_size_check(image2)

# # Сохранение изображений
# output_image1_path = "output_image1.jpg" 
# output_image2_path = "output_image2.jpg"  

# # Сохраняем изображения
# cv2.imwrite(output_image1_path, image1)
# cv2.imwrite(output_image2_path, image2)


# Отображение результатов
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Первая деталь")
plt.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Вторая деталь")
plt.imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.tight_layout()
plt.show()


