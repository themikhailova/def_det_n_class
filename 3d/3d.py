import numpy as np
import trimesh
from PIL import Image, ImageOps
import io
import cv2

# Параметры и пути к файлам
obj_file = 'rotated_model_v1.obj'  # Путь к .obj модели
mask_output_file = 'output.jpg'  # Путь для сохранения маски
input_image_path_back = '9.jpg'  # Входное изображение детали (фон)
output_image_path = 'det_masked_output.jpg'  # Путь для сохранения результата
input_image_path = 'det_orig.jpg'  # Входное изображение

def align_angle(image, output_path=None, show_result=False):
    """
    Автоматическое выравнивание изображения по углу наклона детали.
    
    :param image: Входное изображение (numpy array).
    :param output_path: Путь для сохранения выровненного изображения (опционально).
    :param show_result: Флаг для отображения результата (по умолчанию False).
    :return: Выровненное изображение.
    """
    # Преобразование в оттенки серого и обработка
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        raise ValueError("Контуры не найдены. Проверьте изображение.")

    # Работа с самым большим контуром
    largest_contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest_contour)
    angle = rect[-1]  # Угол наклона
    print(f"Найден угол: {angle}")

    # Вычисляем параметры поворота
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle - 90, 1.0)
    rotated = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    if output_path:
        cv2.imwrite(output_path, rotated)

    if show_result:
        cv2.imshow("Aligned Image", rotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return rotated

def determine_background(image):
    """
    Определяет тип фона изображения (светлый или тёмный).
    
    :param image: Исходное изображение (numpy array).
    :return: "light" или "dark".
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    mean_brightness = np.mean(gray_image)
    print(f"Средняя яркость: {mean_brightness}")
    return "light" if mean_brightness > 127 else "dark"

def crop_to_content(image):
    """
    Обрезка изображения по содержимому (убираем лишний фон).
    
    :param image: Входное изображение (numpy array).
    :return: Обрезанное изображение.
    """
    coords = cv2.findNonZero(image)
    x, y, w, h = cv2.boundingRect(coords)
    return image[y:y + h, x:x + w]

def crop_to_contour_img(image, contour):
    """
    Обрезает изображение по ограничивающему прямоугольнику заданного контура.
    
    :param image: Входное изображение (numpy array).
    :param contour: Контур объекта.
    :return: Обрезанное изображение.
    """
    x, y, contour_width, contour_height = cv2.boundingRect(contour)
    return image[y:y + contour_height, x:x + contour_width]

# Шаг 1. Загрузка 3D-модели и создание маски
mesh = trimesh.load(obj_file)
if mesh.is_empty:
    raise ValueError("Модель пуста.")

print("Границы оригинальной модели:", mesh.bounds)

scene = trimesh.Scene(mesh)
scene.camera.resolution = (512, 512)
scene.camera.fov = (90, 90)

# Устанавливаем камеру
min_bound, max_bound = mesh.bounds
center_point = (min_bound + max_bound) / 2
scene.camera.look_at([center_point], distance=2)

# Рендерим изображение модели
image_data = scene.save_image(background=[0, 0, 0, 255])
image = Image.open(io.BytesIO(image_data)).convert("L")
binary_image = image.point(lambda x: 255 if x > 1 else 0, mode='1')
binary_image.save(mask_output_file, format="JPEG")
print(f"Маска сохранена как {mask_output_file}")

# Шаг 2. Центрирование и работа с изображением
mask_image = cv2.imread(mask_output_file, cv2.IMREAD_GRAYSCALE)
mask_image = crop_to_content(mask_image)
cv2.imwrite('mask_croped.jpg', mask_image)

image = cv2.imread(input_image_path)
image = align_angle(image, 'rotated_orig_cont.jpg')
image_back = cv2.imread(input_image_path_back)
image_back = align_angle(image_back, 'rotated_orig.jpg')

# Шаг 1. Центрирование изображения и маски
image = cv2.imread(input_image_path)
image = align_angle(image, 'rotated_orig_cont.jpg')  # Центрирование изображения для наложения маски

image_back = cv2.imread(input_image_path_back)
image_back = align_angle(image_back, 'rotated_orig.jpg')  # Центрирование фона

# Определение типа фона
background_type = determine_background(image)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
inverted = cv2.bitwise_not(gray) if background_type == "light" else gray
_, binary = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Поиск контуров для центрирования маски
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if contours:
    # Самый крупный контур на изображении
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, contour_width, contour_height = cv2.boundingRect(largest_contour)

    # Обрезка изображения по основному контуру
    centered_image = crop_to_contour_img(image, largest_contour)
    centered_image_back = crop_to_contour_img(image_back, largest_contour)

    # Масштабирование маски до размеров обрезанного изображения
    resized_mask = cv2.resize(mask_image, (contour_width, contour_height), interpolation=cv2.INTER_AREA)
    _, resized_mask = cv2.threshold(resized_mask, 127, 255, cv2.THRESH_BINARY)

    # Центрирование маски на изображении
    h, w = centered_image.shape[:2]
    mask_full = np.zeros((h, w), dtype=np.uint8)
    mask_h, mask_w = resized_mask.shape
    center_x, center_y = (w // 2, h // 2)
    mask_x, mask_y = (mask_w // 2, mask_h // 2)

    # Вычисление смещения маски относительно центра изображения
    top_left_x = max(0, center_x - mask_x)
    top_left_y = max(0, center_y - mask_y)

    # Накладываем маску
    mask_full[top_left_y:top_left_y + mask_h, top_left_x:top_left_x + mask_w] = resized_mask
    result = cv2.bitwise_and(centered_image_back, centered_image_back, mask=mask_full)

    # Сохраняем результат
    cv2.imwrite(output_image_path, result)
    print(f"Изображение с наложенной маской сохранено как {output_image_path}")

    # Для отладки
    cv2.imshow("Centered Image", centered_image)
    cv2.imshow("Mask Full", mask_full)
    cv2.imshow("Result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Контур на изображении не найден.")
