import trimesh
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import io
from skimage.feature import hog
from skimage import exposure

input_obj_file = 'details_v1.obj'
output_mask_file = 'mask.jpg'
input_image_path = 'det_orig.jpg'

def compare_sift(image1, image2):
    # Преобразуем в серые
    

    if len(image1.shape) == 3:  # Если 3 канала (RGB)
        gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)  # Преобразуем в оттенки серого
    elif len(image1.shape) == 2:  # Если уже одноцветное изображение (оттенки серого)
        gray1 = image1  # Используем изображение как есть
    else:
        raise ValueError("Неверное количество каналов в изображении")
    
    if len(image2.shape) == 3:  # Если 3 канала (RGB)
        gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)  # Преобразуем в оттенки серого
    elif len(image2.shape) == 2:  # Если уже одноцветное изображение (оттенки серого)
        gray2 = image2  # Используем изображение как есть
    else:
        raise ValueError("Неверное количество каналов в изображении")

    sift = cv2.SIFT_create()

    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    # Сравниваем дескрипторы с помощью BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    if des1 is not None:
        if des2 is not None:
            matches = bf.match(des1, des2)
            len_matches = len(matches)
        else:
            len_matches = 0
    else:
        len_matches = 0
    
    # img_matches = cv2.drawMatches(image1, kp1, image2, kp2, matches, None)
    # plt.imshow(img_matches)
    # plt.show()

    return len_matches

# def extract_hog_features(image, target_size=(128,128)):
#     """
#     Извлекает признаки HOG из изображения.
#     :param image: Входное изображение (numpy array).
#     :return: Признаки HOG.
#     """
#     # Приводим изображение к одинаковому размеру
#     image_resized = cv2.resize(image, target_size)
#     # Проверяем, если изображение уже в оттенках серого
#     if len(image.shape) == 3:  # Если 3 канала (RGB)
#         gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Преобразуем в оттенки серого
#     elif len(image.shape) == 2:  # Если уже одноцветное изображение (оттенки серого)
#         gray_image = image  # Используем изображение как есть
#     else:
#         raise ValueError("Неверное количество каналов в изображении")
#     features, hog_image = hog(gray_image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
#     hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))  # Улучшаем визуализацию
#     return features, hog_image_rescaled

# # Функция для вычисления разницы между двумя гистограммами HOG
# def compare_hog_features(features1, features2):
#     """
#     Сравнивает два набора признаков HOG.
#     :param features1: Признаки HOG для первого изображения.
#     :param features2: Признаки HOG для второго изображения.
#     :return: Степень различия между признаками (меньше — лучше).
#     """
#     distance = np.linalg.norm(features1 - features2)  # Используем евклидово расстояние
#     return distance

def compare_contours(c1, c2):
    """
    Сравнивает два контура с использованием cv2.matchShapes.
    :param c1: Первый контур.
    :param c2: Второй контур.
    :return: Степень различия между контурами.
    """
    # Проверяем, что контуры содержат достаточно точек
    if len(c1) < 5 or len(c2) < 5:
        print(f"Ошибка: Один из контуров слишком мал. c1: {len(c1)}, c2: {len(c2)}")
        return float('inf')  # Возвращаем "максимальную разницу"

    # Проверяем масштабы контуров и нормализуем их
    c1 = np.array(c1, dtype=np.float32)
    c2 = np.array(c2, dtype=np.float32)

    if np.max(c1) > 0:
        c1 /= np.max(c1)
    if np.max(c2) > 0:
        c2 /= np.max(c2)

    # Сравнение контуров
    try:
        difference = cv2.matchShapes(c1, c2, cv2.CONTOURS_MATCH_I1, 0.0)
    except cv2.error as e:
        print(f"Ошибка при сравнении контуров: {e}")
        return float('inf')  # Возвращаем "максимальную разницу"

    return difference

# Функция для обрезки изображения маски
def crop_to_content(image):
    image_uint8 = (image * 255).astype(np.uint8)
    coords = cv2.findNonZero(image_uint8)  # Ненулевые пиксели
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)  # Ограничивающий прямоугольник
        cropped_image = image_uint8[y:y+h, x:x+w]
        return cropped_image
    else:
        return image_uint8

# Масштабирование маски до размеров целевого контура
def resize_mask_to_contour(mask, target_contour):
    x, y, contour_width, contour_height = cv2.boundingRect(target_contour)
    resized_mask = cv2.resize(mask, (contour_width, contour_height), interpolation=cv2.INTER_AREA)
    _, resized_mask = cv2.threshold(resized_mask, 127, 255, cv2.THRESH_BINARY)
    return resized_mask

# # Функция для комбинированного сравнения
# def combined_comparison(image_real, image_model, contour_real, contour_model):
#     # Сравниваем контуры
#     contour_difference = compare_contours(contour_real, contour_model)

#     # Извлекаем HOG признаки для обоих изображений
#     real_hog_features, _ = extract_hog_features(image_real)
#     model_hog_features, _ = extract_hog_features(image_model)

#     sift_comp = compare_sift(image_real, image_model)

#     # Сравниваем HOG признаки
#     hog_difference = compare_hog_features(real_hog_features, model_hog_features)
#     print('hog: ', hog_difference)
#     print('contour: ', contour_difference)
#     print('sift: ', sift_comp)
#     # Общая разница: комбинируем контуры и HOG с одинаковыми весами
#     total_difference = contour_difference + hog_difference
#     return total_difference



def look_at(camera_position, target_position, up_vector):
    # Направление "вперёд" (от камеры к цели)
    forward = np.array(target_position) - np.array(camera_position)
    forward = forward / np.linalg.norm(forward)  # Нормализация

    # Направление "вправо" (перпендикуляр к up и forward)
    right = np.cross(forward, up_vector)
    right = right / np.linalg.norm(right)

    # Новое направление "вверх" (перпендикуляр к forward и right)
    up = np.cross(right, forward)

    # Матрица камеры (4x4)
    camera_matrix = np.eye(4)
    camera_matrix[:3, 0] = right
    camera_matrix[:3, 1] = up
    camera_matrix[:3, 2] = -forward
    camera_matrix[:3, 3] = camera_position

    return camera_matrix

# Основная логика вращения
def rotation(angle_x, angle_y, angle_z, mesh, largest_contour, target_binary, best_rotation, sift_max):
    angle_x_rad = np.radians(angle_x)
    angle_y_rad = np.radians(angle_y)
    angle_z_rad = np.radians(angle_z)

    # Создаем матрицы поворота
    rotation_matrix_x = trimesh.transformations.rotation_matrix(angle_x_rad, [1, 0, 0], mesh.centroid)
    rotation_matrix_y = trimesh.transformations.rotation_matrix(angle_y_rad, [0, 1, 0], mesh.centroid)
    rotation_matrix_z = trimesh.transformations.rotation_matrix(angle_z_rad, [0, 0, 1], mesh.centroid)

    combined_rotation_matrix = np.dot(np.dot(rotation_matrix_x, rotation_matrix_y), rotation_matrix_z)
    rotated_mesh = mesh.copy()
    rotated_mesh.apply_transform(combined_rotation_matrix)
    # Создаём сцену
    scene = trimesh.Scene(rotated_mesh)
    scene.camera.resolution = (256, 256)
    scene.camera.fov = (90, 90)
    # Получаем размер модели и её центр
    bounds = rotated_mesh.bounds  # Границы модели (min, max)
    center = rotated_mesh.center_mass  # Центр модели
    size = np.linalg.norm(bounds[1] - bounds[0])  # Размер модели

    # Расположение камеры
    camera_distance = size 
    camera_position = [camera_distance, camera_distance, camera_distance]
    target_position = center  # Камера смотрит на центр модели
    up_vector = [0, 1, 0]     # Направление вверх

    # Рассчитываем матрицу камеры
    camera_transform = look_at(camera_position, target_position, up_vector)

    # Применяем трансформацию камеры
    scene.camera_transform = camera_transform

    # Рендерим изображение
    # scene = trimesh.Scene(rotated_mesh)
    # scene.camera.resolution = (128, 128)
    # scene.camera.fov = (90, 90)
    
    # min_bound, max_bound = rotated_mesh.bounds
    # center_point = (min_bound + max_bound) / 2
    # scene.camera.look_at([center_point], distance=2)
    # print(center_point)

    image_data = scene.save_image(background=[0, 0, 0, 255])
    image = Image.open(io.BytesIO(image_data))

    if largest_contour is not None:
        gray_image = image.convert("L")
        binary_image = np.array(gray_image.point(lambda x: 255 if x > 1 else 0, mode='1'))
        cropped_mask = crop_to_content(binary_image)

        resized_mask = resize_mask_to_contour(cropped_mask, largest_contour)
        cropped_target = crop_to_contour(target_binary, largest_contour)

        # mask_contours, _ = cv2.findContours(resized_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # target_contours, _ = cv2.findContours(cropped_target, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Выполняем комбинированное сравнение (контуры + HOG)
        # difference = combined_comparison(cropped_target, resized_mask, mask_contours[0], largest_contour)

        # if difference < min_difference:
        #     min_difference = difference
        #     best_rotation = (angle_x, angle_y, angle_z)

        sift_comp = compare_sift(cropped_target, resized_mask)

        if sift_max < sift_comp:
            sift_max = sift_comp
            print(sift_max, sift_comp)
            best_rotation = (angle_x, angle_y, angle_z)
            
        print(f"Поворот: ({angle_x}, {angle_y}, {angle_z}) градусов, Разница: {sift_comp}")

    return best_rotation, sift_max

def crop_to_contour(image, contour):
    """
    Обрезает изображение по ограничивающему прямоугольнику заданного контура.
    :param image: Исходное изображение (numpy array).
    :param contour: Контур объекта.
    :return: Обрезанное изображение.
    """
    # Получаем ограничивающий прямоугольник
    x, y, contour_width, contour_height = cv2.boundingRect(contour)
    cropped_image = image[y:y+contour_height, x:x+contour_width]
    return cropped_image

def align_angle(image, output_path=None, show_result=False):
    """
    Автоматическое выравнивание детали на изображении, учитывая небольшой наклон.
    
    :param image_path: Путь к входному изображению
    :param output_path: Путь для сохранения выровненного изображения (необязательно)
    :param show_result: Флаг для отображения результата (по умолчанию False)
    :return: Выровненное изображение
    """
    
    # Проверяем, если изображение уже в оттенках серого
    if len(image.shape) == 3:  # Если 3 канала (RGB)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Преобразуем в оттенки серого
    elif len(image.shape) == 2:  # Если уже одноцветное изображение (оттенки серого)
        gray = image  # Используем изображение как есть
    else:
        raise ValueError("Неверное количество каналов в изображении")
    
    # Размытие для удаления шума
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Выделение контуров
    edges = cv2.Canny(blurred, 50, 150)
    
    # Нахождение контуров
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        raise ValueError("Контуры не найдены. Проверьте изображение.")
    
    # Выбор самого большого контура
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Вычисление минимального прямоугольника
    rect = cv2.minAreaRect(largest_contour)
    angle = rect[-1]
    print(angle)
    # Корректировка угла
    # if angle < -45:
    #     angle += 90
    
    # Центр изображения для поворота
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    
    # Поворот изображения
    rotation_matrix = cv2.getRotationMatrix2D(center, angle-90, 1.0)
    rotated = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    # Сохранение результата
    if output_path:
        cv2.imwrite(output_path, rotated)
    
    # Отображение результата (если требуется)
    if show_result:
        cv2.imshow("Aligned Image", rotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return rotated

def rotate_6sides(directions, mesh, largest_contour, target_binary,best_rotation, sift_max):
    for angle_x, angle_y, angle_z in directions:
        # try:
            best_rotation, sift_max = rotation(angle_x, angle_y, angle_z, mesh, largest_contour, target_binary, best_rotation, sift_max)
        # except Exception as e:
        #     print(f"Ошибка при обработке углов ({angle_x}, {angle_y}, {angle_z}): {e}")
    return best_rotation, sift_max

def min_rotate(angle_x, angle_y, angle_z, n, m, mesh,best_rotation, target_binary, sift_max):
    start_x = angle_x-n
    start_y = angle_y-n
    start_z = angle_z-n
    end_x = angle_x+n
    end_y = angle_y+n
    end_z = angle_z+n
    for angle_x in range(start_x, end_x, m):  # Поворот по оси X
        for angle_y in range(start_y, end_y, m):  # Поворот по оси Y
            # for angle_z in range(start_z, end_z, m):  # Поворот по оси Z
                try:
                    best_rotation, sift_max = rotation(angle_x, angle_y, angle_z, mesh, largest_contour, target_binary, best_rotation, sift_max)
                except Exception as e:
                    print(f"Ошибка при обработке углов ({angle_x}, {angle_y}, {angle_z}): {e}")

    return best_rotation, sift_max

# Запуск основного кода
mesh = trimesh.load(input_obj_file)
directions = [(0, 0, 0), (180, 0, 0), (90, 0, 0), (270, 0, 0), (0, 90, 0), (0, -90, 0)]
# directions = [(0, 90, 0)]

# min_difference = float('inf')
sift_max = 0
best_rotation = None

target_image = cv2.imread(input_image_path, cv2.IMREAD_ANYCOLOR)
target_image_rot = align_angle(target_image, './rot_det.jpg', True)
target_image = cv2.imread('./rot_det.jpg', cv2.IMREAD_GRAYSCALE)

_, target_binary = cv2.threshold(target_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
contours, _ = cv2.findContours(target_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if contours:
    largest_contour = max(contours, key=cv2.contourArea)

best_rotation, sift_max = rotate_6sides(directions, mesh, largest_contour, target_binary, best_rotation, sift_max)

print(f"Лучший угол поворота из 6 сторон: {best_rotation}, Схожесть: {sift_max}")

angle_x, angle_y, angle_z = best_rotation
best_rotation, sift_max = min_rotate(angle_x, angle_y, angle_z, 15, 4, mesh, best_rotation, target_binary, sift_max)

print(f"Лучший угол поворота: {best_rotation}, Схожесть: {sift_max}")
angle_x, angle_y, angle_z = best_rotation
print(angle_x, angle_y, angle_z)
angle_x_rad = np.radians(angle_x)
angle_y_rad = np.radians(angle_y)
angle_z_rad = np.radians(angle_z)

rotation_matrix_x = trimesh.transformations.rotation_matrix(angle_x_rad, [1, 0, 0], mesh.centroid)
rotation_matrix_y = trimesh.transformations.rotation_matrix(angle_y_rad, [0, 1, 0], mesh.centroid)
rotation_matrix_z = trimesh.transformations.rotation_matrix(angle_z_rad, [0, 0, 1], mesh.centroid)

combined_rotation_matrix = np.dot(np.dot(rotation_matrix_x, rotation_matrix_y), rotation_matrix_z)
rotated_mesh = mesh.copy()
rotated_mesh.apply_transform(combined_rotation_matrix)

# Сохранение повернутой модели
output_obj_file_rotated = 'rotated_model.obj'
rotated_mesh.export(output_obj_file_rotated)
print(f"Повернутая модель сохранена как {output_obj_file_rotated}")

# Визуализация повернутой модели
rotated_mesh.show()  # Открывает 3D-просмотрщик Trimesh

