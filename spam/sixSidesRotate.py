import trimesh
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import io
from skimage.feature import hog
from skimage import exposure

# Шаг 1: Загрузка модели
input_obj_file = './3d/detail_3.obj'
output_mask_file = 'mask.jpg'
input_image_path = './3d/32noback.jpg'

def to_gray(image):
    if len(image.shape) == 3:  # Если 3 канала (RGB)
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Преобразуем в оттенки серого
    elif len(image.shape) == 2:  # Если уже одноцветное изображение (оттенки серого)
        return image  # Используем изображение как есть
    else:
        raise ValueError("Неверное количество каналов в изображении")

def compare_sift(image1, image2):
    gray1 = to_gray(image1)
    gray2 = to_gray(image2)

    sift = cv2.SIFT_create()

    # дескрипторы
    _, des1 = sift.detectAndCompute(gray1, None)
    _, des2 = sift.detectAndCompute(gray2, None)

    # сравнение BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    if des1 is not None:
        if des2 is not None:
            matches = bf.match(des1, des2)
            len_matches = len(matches)
        else:
            len_matches = 0
    else:
        len_matches = 0
    
    # Визуализируем совпадения
    # img_matches = cv2.drawMatches(image1, kp1, image2, kp2, matches, None)
    # plt.imshow(img_matches)
    # plt.show()

    return len_matches

def compare_contours(contour1, contour2):
    '''
    Сравнивает два контура с использованием cv2.matchShapes
    :param contour1: Первый контур
    :param contour2: Второй контур
    :return: различие между контурами
    '''
    if len(contour1) < 5 or len(contour2) < 5:
        print(f"Ошибка: Один из контуров слишком мал. contour1: {len(contour1)}, contour2: {len(contour2)}")
        return float('inf')  

    contour1 = np.array(contour1, dtype=np.float32)
    contour2 = np.array(contour2, dtype=np.float32)

    if np.max(contour1) > 0:
        contour1 /= np.max(contour1)
    if np.max(contour2) > 0:
        contour2 /= np.max(contour2)

    try:
        difference = cv2.matchShapes(contour1, contour2, cv2.CONTOURS_MATCH_I1, 0.0)
    except cv2.error as e:
        print(f"Ошибка при сравнении контуров: {e}")
        return float('inf')  

    return difference

# функция для обрезки изображения маски
def crop_to_content(image):
    image_uint8 = (image * 255).astype(np.uint8)
    coords = cv2.findNonZero(image_uint8)  # ненулевые пиксели
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)  
        cropped_image = image_uint8[y:y+h, x:x+w]
        return cropped_image
    else:
        return image_uint8
    
def crop_to_contour(image, contour):
    '''
    Обрезает изображение по ограничивающему прямоугольнику заданного контура
    :param image: Исходное изображение
    :param contour: Контур объекта
    :return: Обрезанное изображение
    '''
    x, y, contour_width, contour_height = cv2.boundingRect(contour)
    cropped_image = image[y:y+contour_height, x:x+contour_width]
    return cropped_image

# масштабирование маски до размеров целевого контура
def resize_mask_to_contour(mask, target_contour):
    _, _, contour_width, contour_height = cv2.boundingRect(target_contour)
    resized_mask = cv2.resize(mask, (contour_width, contour_height), interpolation=cv2.INTER_AREA)
    _, resized_mask = cv2.threshold(resized_mask, 127, 255, cv2.THRESH_BINARY)
    return resized_mask

# функция для сравнения контуров (признаков)
def combined_comparison(image_real, image_model, contour_real, contour_model):
    # контуры
    contour_difference = compare_contours(contour_real, contour_model)
    # признаки
    sift_comp = compare_sift(image_real, image_model)
    print('contour: ', contour_difference)
    print('sift: ', sift_comp)
    # return contour_difference
    return contour_difference

# настройка ракурса камеры
def look_at(camera_position, target_position, up_vector):
    # "вперёд" (от камеры к цели)
    forward = np.array(target_position) - np.array(camera_position)
    forward = forward / np.linalg.norm(forward)  # Нормализация

    # "вправо" (перпендикуляр к up и forward)
    right = np.cross(forward, up_vector)
    right = right / np.linalg.norm(right)

    # новое направление "вверх" (перпендикуляр к forward и right)
    up = np.cross(right, forward)

    # матрица камеры (4x4)
    camera_matrix = np.eye(4)
    camera_matrix[:3, 0] = right
    camera_matrix[:3, 1] = up
    camera_matrix[:3, 2] = -forward
    camera_matrix[:3, 3] = camera_position

    return camera_matrix

# вращение
def rotation(angle_x, angle_y, angle_z, mesh, largest_contour, target_binary, best_rotation, sift_max):
    angle_x_rad = np.radians(angle_x)
    angle_y_rad = np.radians(angle_y)
    angle_z_rad = np.radians(angle_z)

    # матрицы поворота
    rotation_matrix_x = trimesh.transformations.rotation_matrix(angle_x_rad, [1, 0, 0], mesh.centroid)
    rotation_matrix_y = trimesh.transformations.rotation_matrix(angle_y_rad, [0, 1, 0], mesh.centroid)
    rotation_matrix_z = trimesh.transformations.rotation_matrix(angle_z_rad, [0, 0, 1], mesh.centroid)

    combined_rotation_matrix = np.dot(np.dot(rotation_matrix_x, rotation_matrix_y), rotation_matrix_z)
    rotated_mesh = mesh.copy()
    rotated_mesh.apply_transform(combined_rotation_matrix)
    # сцена
    scene = trimesh.Scene(rotated_mesh)
    scene.camera.resolution = (512, 512)
    scene.camera.fov = (90, 90)
    # размер модели и её центр
    bounds = rotated_mesh.bounds  # границы модели (min, max)
    center = rotated_mesh.center_mass  # центр модели
    size = np.linalg.norm(bounds[1] - bounds[0])  # размер модели

    # расположение камеры
    camera_distance = size 
    camera_position = [camera_distance, camera_distance, camera_distance]
    target_position = center  # камера смотрит на центр модели
    up_vector = [0, 1, 0]     # направление вверх

    # матрица камеры
    camera_transform = look_at(camera_position, target_position, up_vector)
    scene.camera_transform = camera_transform
    
    # рендер
    scene = trimesh.Scene(rotated_mesh)
    scene.camera.resolution = (1024, 1024)
    scene.camera.fov = (90, 90)
    
    min_bound, max_bound = rotated_mesh.bounds
    center_point = (min_bound + max_bound) / 2
    scene.camera.look_at([center_point], distance=2)
    print(center_point)
    image_data = scene.save_image(background=[0, 0, 0, 255])
    image = Image.open(io.BytesIO(image_data))

    if largest_contour is not None:
        gray_image = image.convert("L")
        binary_image = np.array(gray_image.point(lambda x: 255 if x > 1 else 0, mode='1'))
        cropped_mask = crop_to_content(binary_image)

        resized_mask = resize_mask_to_contour(cropped_mask, largest_contour)
        cropped_target = crop_to_contour(target_binary, largest_contour)

        mask_contours, _ = cv2.findContours(resized_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # сравнение
        difference = combined_comparison(cropped_target, resized_mask, mask_contours[0], largest_contour)

        if difference < sift_max:
            sift_max = difference
            best_rotation = (angle_x, angle_y, angle_z)

        # sift_comp = compare_sift(cropped_target, resized_mask)

        # if sift_max < sift_comp:
        #     sift_max = sift_comp
        #     print(sift_max, sift_comp)
        #     best_rotation = (angle_x, angle_y, angle_z)
            
        print(f"Поворот: ({angle_x}, {angle_y}, {angle_z}) градусов, Схожесть: {difference}")

    return best_rotation, sift_max

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
    # start_z = angle_z-n
    end_x = angle_x+n
    end_y = angle_y+n
    # end_z = angle_z+n
    for angle_x in range(start_x, end_x, m):  # Поворот по оси X
        for angle_y in range(start_y, end_y, m):  # Поворот по оси Y
            # for angle_z in range(start_z, end_z, m):  # Поворот по оси Z
                try:
                    best_rotation, sift_max = rotation(angle_x, angle_y, angle_z, mesh, largest_contour, target_binary, best_rotation, sift_max)
                except Exception as e:
                    print(f"Ошибка при обработке углов ({angle_x}, {angle_y}, {angle_z}): {e}")

    return best_rotation, sift_max


mesh = trimesh.load(input_obj_file)
directions = [(0, 0, 0), (180, 0, 0), (90, 0, 0), (270, 0, 0), (0, 90, 0), (0, -90, 0)]

# min_difference = float('inf')
sift_max = float('inf')
best_rotation = None

target_image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
cv2.imshow('target', target_image)
cv2.waitKey(0)
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

    output_obj_file_rotated = './3d/rotated_model_v3.obj'
    rotated_mesh.export(output_obj_file_rotated)
    print(f"Повернутая модель: {output_obj_file_rotated}")

    rotated_mesh.show()

