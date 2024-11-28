import numpy as np
import trimesh
from PIL import Image, ImageOps
import io
import cv2

obj_file = 'rotated_model.obj'  # путь к .obj модели
mask_output_file = 'output.jpg'  # путь для сохранения маски
input_image_path_back = '9.jpg'  # входное изображение детали (фон)
output_image_path = 'det_masked_output.jpg'  # путь для сохранения результата
input_image_path = 'det_orig.jpg'  # входное изображение

def align_angle(image, output_path=None, show_result=False, angle=None):
    """
    выравнивание изображения по углу наклона детали
    
    :param image: Входное изображение (numpy)
    :param output_path: Путь для сохранения выровненного изображения 
    :param show_result: Флаг для отображения результата
    :return: Выровненное изображение 
    """
    # Проверяем, если изображение уже в оттенках серого
    if len(image.shape) == 3:  # Если 3 канала (RGB)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Преобразуем в оттенки серого
    elif len(image.shape) == 2:  # Если уже одноцветное изображение (оттенки серого)
        gray = image  # Используем изображение как есть
    else:
        raise ValueError("Неверное количество каналов в изображении")
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        raise ValueError("Контуры не найдены")

    largest_contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest_contour)
    if angle==None:
        angle = rect[-1]  # угол наклона
        print(f"Найден угол: {angle}")

    # параметры поворота
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle - 90, 1.0)
    rotated = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    if output_path:
        cv2.imwrite(output_path, rotated)

    if show_result:
        cv2.imshow("Выровненное фото", rotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return rotated, angle if angle else 90

def determine_background(image):
    """
    тип фона изображения (светлый или тёмный)
    
    :param image: исходное изображение (numpy)
    :return: "light" или "dark"
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    mean_brightness = np.mean(gray_image)
    print(f"Средняя яркость: {mean_brightness}")
    return "light" if mean_brightness > 127 else "dark"

def crop_to_content(image):
    """
    обрезка маски по содержимому (убираем лишний фон)
    
    :param image: входное изображение (numpy)
    :return: обрезанное изображение
    """
    coords = cv2.findNonZero(image)
    x, y, w, h = cv2.boundingRect(coords)
    # x, y, w, h = x+4, y+4, w-4, h-4
    return image[y:y + h, x:x + w]

def crop_to_contour_img(image, contour):
    """
    обрезка изображения по ограничивающему прямоугольнику контура
    
    :param image: входное изображение (numpy)
    :param contour: контур объекта
    :return: обрезанное изображение
    """
    x, y, contour_width, contour_height = cv2.boundingRect(contour)
    return image[y:y + contour_height, x:x + contour_width]

# сглаживание краев маски
def smooth_mask_strong(mask, kernel_size=21, sigma=10.0, threshold_value=200):
    """
    сглаживает края маски для получения более ровных границ
    
    :param mask: входная маска (numpy)
    :param kernel_size: размер ядра для размытия
    :param sigma: параметр размытия для гауссова фильтра
    :param threshold_value: пороговое значение для бинаризации
    :return: сглаженная маска
    """
    # увеличиние маски (распространение белой области)
    dilated_mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)))
    # размытие краев
    blurred_mask = cv2.GaussianBlur(dilated_mask, (kernel_size, kernel_size), sigma)
    # бинаризация для создания резких краёв
    _, smoothed_mask = cv2.threshold(blurred_mask, threshold_value, 255, cv2.THRESH_BINARY)
    return smoothed_mask

def stretch_mask_vertically(mask):
    """
    Растягивает белую область на маске вверх, чтобы убрать зазоры сверху.
    
    :param mask: Входная маска (numpy array, dtype=uint8).
    :return: Маска с растянутой белой областью вверх.
    """
    # Найти контуры объекта
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("Объект не найден на маске.")
    
    # Найти ограничивающий прямоугольник вокруг самого большого контура
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    
    # Обрезать белую область объекта
    cropped_mask = mask[y:y+h, x:x+w]
    
    # Растянуть белую область на всю высоту сверху
    stretched_mask = cv2.resize(cropped_mask, (w, mask.shape[0] - y), interpolation=cv2.INTER_AREA)
    
    # Создать новую маску
    final_mask = np.zeros_like(mask, dtype=np.uint8) 
    
    # Вставить растянутую белую область
    final_mask[:mask.shape[0] - y, x:x+w] = stretched_mask
    # Найти ограничивающий прямоугольник вокруг объекта
    # x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    
    # # Обрезать белую область объекта
    # cropped_mask = mask[y:y+h, x:x+w]
    
    # # Растянуть белую область на всю высоту исходного изображения
    # stretched_mask = cv2.resize(cropped_mask, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_AREA)
    
    # Создать пустую маску и вставить растянутую область
    #final_mask = np.zeros_like(mask, dtype=np.uint8)
    # final_mask[:, x:x+w] = stretched_mask
    
    
    return final_mask

# def look_at(camera_position, target_position, up_vector):
#     # Направление "вперёд" (от камеры к цели)
#     forward = np.array(target_position) - np.array(camera_position)
#     forward = forward / np.linalg.norm(forward)  # Нормализация

#     # Направление "вправо" (перпендикуляр к up и forward)
#     right = np.cross(forward, up_vector)
#     right = right / np.linalg.norm(right)

#     # Новое направление "вверх" (перпендикуляр к forward и right)
#     up = np.cross(right, forward)

#     # Матрица камеры (4x4)
#     camera_matrix = np.eye(4)
#     camera_matrix[:3, 0] = right
#     camera_matrix[:3, 1] = up
#     camera_matrix[:3, 2] = -forward
#     camera_matrix[:3, 3] = camera_position

#     return camera_matrix


# загрузка 3D-модели и создание маски
mesh = trimesh.load(obj_file)
if mesh.is_empty:
    raise ValueError("Модель пуста.")

print("Границы оригинальной модели:", mesh.bounds)

scene = trimesh.Scene(mesh)
scene.camera.resolution = (512, 512)
scene.camera.fov = (90, 90)
# mesh.show() 
# min_bound, max_bound = mesh.bounds
# center_point = (min_bound + max_bound) / 2
# scene.camera.look_at([center_point], distance=2)
# Получаем размер модели и её центр



image_data = scene.save_image(background=[0, 0, 0, 255])
image = Image.open(io.BytesIO(image_data)).convert("L")
binary_image = image.point(lambda x: 255 if x > 1 else 0, mode='1')
binary_image.save(mask_output_file, format="JPEG")
print(f"Маска сохранена как {mask_output_file}")

mask_image = cv2.imread(mask_output_file, cv2.IMREAD_GRAYSCALE)

mask_image = crop_to_content(mask_image)
cv2.imwrite('mask_croped.jpg', mask_image)


image = cv2.imread(input_image_path)
image, angle1 = align_angle(image, 'rotated_orig_cont.jpg')
image_back = cv2.imread(input_image_path_back)
image_back, angle1 = align_angle(image_back, 'rotated_orig.jpg', angle=angle1)

# определение типа фона
background_type = determine_background(image)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
inverted = cv2.bitwise_not(gray) if background_type == "light" else gray
_, binary = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# поиск контуров для центрирования маски
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if contours:
    # самый крупный контур
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, contour_width, contour_height = cv2.boundingRect(largest_contour)

    # обрезка изображения по контуру
    centered_image = crop_to_contour_img(image, largest_contour)
    centered_image_back = crop_to_contour_img(image_back, largest_contour)
    
    # масштабирование маски до размеров обрезанного изображения
    resized_mask = cv2.resize(mask_image, (contour_width+40, contour_height), interpolation=cv2.INTER_AREA)

    # resized_mask = crop_to_contour_img(resized_mask, largest_contour)
    _, resized_mask = cv2.threshold(resized_mask, 127, 255, cv2.THRESH_BINARY)
    resized_mask_contours, _ = cv2.findContours(resized_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if resized_mask_contours:
        largest_contour_mask = max(resized_mask_contours, key=cv2.contourArea)
        resized_mask = crop_to_contour_img(resized_mask, largest_contour_mask)
    
    # cv2.imshow("mask", mask_image)
    cv2.imshow("resized_mask", resized_mask)
    cv2.waitKey(0)
    #resized_mask = stretch_mask_vertically(resized_mask)
    # сглаживание краёв маски
    # smoothed_mask = smooth_mask_strong(resized_mask)
    smoothed_mask = resized_mask
    # центрирование маски на изображении
    h, w = centered_image.shape[:2]
    mask_full = np.zeros((h, w), dtype=np.uint8)
    mask_h, mask_w = smoothed_mask.shape
    center_x, center_y = (w // 2, h // 2)
    mask_x, mask_y = (mask_w // 2, mask_h // 2)
    
    # вычисление смещения маски относительно центра изображения
    top_left_x = max(0, center_x - mask_x)
    top_left_y = max(0, center_y - mask_y-30)
    print(top_left_x, top_left_y)
    # наложение сглаженной маски
    mask_full[top_left_y:top_left_y + mask_h, top_left_x:top_left_x + mask_w] = smoothed_mask
    result = cv2.bitwise_and(centered_image_back, centered_image_back, mask=mask_full)

    cv2.imwrite(output_image_path, result)
    print(f"Фото с маской на пути: {output_image_path}")

    # Для отладки
    cv2.imshow("centered image", centered_image)
    cv2.imshow("mask", mask_full)
    cv2.imshow("result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Контур на изображении не найден")
