import numpy as np
import trimesh
from PIL import Image, ImageOps
import io
import cv2

obj_file = 'rotated_model.obj'  # Путь к .obj модели
mask_output_file = 'output.jpg'  # Путь для сохранения маски
input_image_path = 'det_ret2.jpg'  # Входное изображение детали
output_image_path = 'det_masked_output.jpg'  # Путь для сохранения результата

def determine_background(image):
    """
    Определяет, светлый или темный фон у изображения, чтобы кореектно находить контур
    :param image: Исходное изображение (numpy array).
    :return: "light" если фон светлый, "dark" если фон темный.
    """
    # Преобразуем изображение в оттенки серого, если оно цветное
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    # Рассчитываем среднюю яркость
    mean_brightness = np.mean(gray_image)
    print(f"Средняя яркость: {mean_brightness}")

    # Порог для определения фона
    return "light" if mean_brightness > 127 else "dark"

# Шаг 1. Загрузка 3D модели и создание маски
mesh = trimesh.load(obj_file)
print(mesh.is_empty)  
print("Границы оригинальной модели:", mesh.bounds)

# Создание сцены и настройка камеры
scene = trimesh.Scene(mesh)
scene.camera.resolution = (512, 512)
scene.camera.fov = (90, 90)

# Рассчитываем центральную точку модели
min_bound, max_bound = mesh.bounds
center_point = (min_bound + max_bound) / 2

# Камера смотрит на объект
scene.camera.look_at([center_point], distance=2)

# Рендерим изображение с черным фоном
image_data = scene.save_image(background=[0, 0, 0, 255])
image = Image.open(io.BytesIO(image_data))

# Конвертация изображения в оттенки серого и применение порога
gray_image = image.convert("L")
binary_image = gray_image.point(lambda x: 255 if x > 1 else 0, mode='1')

# Сохранение маски
binary_image.save(mask_output_file, format="JPEG")
print(f"Маска сохранена как {mask_output_file}")

# Загрузка и обрезка маски по содержимому, чтобы фона был минимум
def crop_to_content(image):
    coords = cv2.findNonZero(image)  # Ненулевые пиксели
    x, y, w, h = cv2.boundingRect(coords)  # Ограничивающий прямоугольник
    cropped_image = image[y:y+h, x:x+w]
    return cropped_image

mask_image = cv2.imread(mask_output_file, cv2.IMREAD_GRAYSCALE)
mask_image = crop_to_content(mask_image)

# Шаг 2. Центрирование объекта на входном изображении
image = cv2.imread(input_image_path)

# Определяем фон изображения
background_type = determine_background(image)

if background_type == "light":
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    inverted = cv2.bitwise_not(gray)
    _, binary = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
else:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


# Поиск контуров
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

if contours:
    # Самый крупный контур
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Ограничивающий прямоугольник детали
    x, y, contour_width, contour_height = cv2.boundingRect(largest_contour)
    
    # Центрирование изображения
    M = cv2.moments(largest_contour)
    if M["m00"] != 0:
        cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
        h, w = gray.shape
        shift_x, shift_y = w // 2 - cx, h // 2 - cy
        
        translation_matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        centered_image = cv2.warpAffine(image, translation_matrix, (w, h))
    else:
        centered_image = image
else:
    print("Контур не найден, изображение не центрировано.")
    centered_image = image

# Преобразуем целевое изображение в оттенки серого
#gray_target = cv2.cvtColor(centered_image, cv2.COLOR_BGR2GRAY)

# Бинаризация целевого изображения
#_, binary_target = cv2.threshold(gray_target, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Поиск контуров в целевом изображении
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if contours:
    # Находим самый крупный контур
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, contour_width, contour_height = cv2.boundingRect(largest_contour)
    # Костыли :(
    contour_width, contour_height = contour_width+10, contour_height+21
    x, y = x-6, y
    # Масштабируем маску по размеру контуров целевого изображения
    # resized_mask = cv2.resize(mask_image, (contour_width, contour_height), interpolation=cv2.INTER_AREA)
    resized_mask = cv2.resize(mask_image, (contour_width, contour_height), interpolation=cv2.INTER_LINEAR)
    _, resized_mask = cv2.threshold(resized_mask, 127, 255, cv2.THRESH_BINARY)
    
    # Создаем пустое изображение для маски
    mask_full = np.zeros_like(gray_target)
    
    # Убедимся, что размеры соответствуют
    mask_full[y:y+contour_height, x:x+contour_width] = resized_mask
    
    # Применяем маску к изображению
    result = cv2.bitwise_and(centered_image, centered_image, mask=mask_full)
    
    # Показываем результат для отладки
    cv2.imshow("Result", result)
    cv2.waitKey(0)

    # Сохраняем результат
    cv2.imwrite(output_image_path, result)
    print(f"Изображение с наложенной маской сохранено как {output_image_path}")
else:
    print("Контур на целевом изображении не найден.")



