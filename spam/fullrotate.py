# тут в общем какой-то бред. лучше смотреть sixSidesRotate!!!

# надо уменьшать количество вычислений: 1) в функцию подавать контур искомой детали а не находить в функции
# 2) попробовать полинин вар: 6 сторон + подгон

import trimesh
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import io

# Шаг 1: Загрузка модели
input_obj_file = 'details_v1.obj'
output_mask_file = 'mask.jpg'
input_image_path = 'det_rot.jpg'

mesh = trimesh.load(input_obj_file)

# n +-, m - шаг
def rot(angle_x, angle_y, angle_z, n, m, mesh, min_difference, best_rotation, target_binary):
    for angle_x in range(angle_x, angle_x+n, m):  # Поворот по оси X
        for angle_y in range(angle_y, angle_y+n, m):  # Поворот по оси Y
            for angle_z in range(angle_z, angle_z+n, m):  # Поворот по оси Z
                # Преобразование углов в радианы
                angle_x_rad = np.radians(angle_x)
                angle_y_rad = np.radians(angle_y)
                angle_z_rad = np.radians(angle_z)

                # Создание матрицы поворота для каждой оси
                rotation_matrix_x = trimesh.transformations.rotation_matrix(angle_x_rad, [1, 0, 0], mesh.centroid)
                rotation_matrix_y = trimesh.transformations.rotation_matrix(angle_y_rad, [0, 1, 0], mesh.centroid)
                rotation_matrix_z = trimesh.transformations.rotation_matrix(angle_z_rad, [0, 0, 1], mesh.centroid)

                # Применяем матрицы поворота
                combined_rotation_matrix = np.dot(np.dot(rotation_matrix_x, rotation_matrix_y), rotation_matrix_z)
                rotated_mesh = mesh
                rotated_mesh.apply_transform(combined_rotation_matrix)

                # Создание сцены и рендеринг
                scene = trimesh.Scene(rotated_mesh)
                scene.camera.resolution = (512, 512)
                scene.camera.fov = (90, 90)

                # Проверка на размеры модели (если модель слишком маленькая)
                min_bound, max_bound = rotated_mesh.bounds
                if np.allclose(min_bound, max_bound):  # Проверка на нулевые размеры
                    print(f"Ошибка: размеры модели слишком малы или нулевые для поворота ({angle_x}, {angle_y}, {angle_z})")
                    continue

                # Находим центр модели
                center_point = (min_bound + max_bound) / 2
                scene.camera.look_at([center_point], distance=2)

                # Проверка, что камера не делит на ноль
                if np.any(np.isnan(center_point)) or np.isinf(center_point[0]) or np.isinf(center_point[1]) or np.isinf(center_point[2]):
                    print(f"Ошибка: некорректная позиция камеры для поворота ({angle_x}, {angle_y}, {angle_z})")
                    continue

                try:
                    # Рендеринг изображения
                    image_data = scene.save_image(background=[0, 0, 0, 255])
                    image = Image.open(io.BytesIO(image_data))

                    # Конвертация в оттенки серого и бинаризация
                    gray_image = image.convert("L")
                    binary_image = np.array(gray_image.point(lambda x: 255 if x > 1 else 0, mode='1'))

                    # Обрезка маски по содержимому
                    cropped_mask = crop_to_content(binary_image)

                    # Поиск самого крупного контура в целевом изображении
                    contours, _ = cv2.findContours(target_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        largest_contour = max(contours, key=cv2.contourArea)

                        # Масштабируем маску до размеров контура
                        resized_mask = resize_mask_to_contour(cropped_mask, largest_contour)

                        # Обрезаем объект в целевом изображении по контуру
                        cropped_target = crop_to_contour(target_binary, largest_contour)

                        # Обрезаем маску аналогично
                        mask_contour = cv2.findContours(resized_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0]
                        cropped_mask_resized = crop_to_contour(resized_mask, mask_contour)

                        # Сравнение обрезанных контуров
                        mask_contours, _ = cv2.findContours(cropped_mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        target_contours, _ = cv2.findContours(cropped_target, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                        difference = 0
                        for c1, c2 in zip(mask_contours, target_contours):
                            difference += cv2.matchShapes(c1, c2, cv2.CONTOURS_MATCH_I1, 0.0)

                        # Проверяем, если текущий угол дает минимальную разницу
                        if difference < min_difference:
                            min_difference = difference
                            best_rotation = (angle_x, angle_y, angle_z)

                        print(f"Поворот: ({angle_x}, {angle_y}, {angle_z}) градусов, Разница: {difference}")
                except ZeroDivisionError as e:
                    print(f"Ошибка деления на ноль при рендеринге ({angle_x}, {angle_y}, {angle_z}): {e}")
                    continue
                except Exception as e:
                    print(f"Ошибка рендеринга для поворота ({angle_x}, {angle_y}, {angle_z}): {e}")
                    continue
    return best_rotation, min_difference

# Функция для обрезки изображения маски
def crop_to_content(image):
    # Преобразуем изображение в формат uint8
    image_uint8 = (image * 255).astype(np.uint8)  # Преобразуем булев массив в [0, 255]
    coords = cv2.findNonZero(image_uint8)  # Ненулевые пиксели
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)  # Ограничивающий прямоугольник
        cropped_image = image_uint8[y:y+h, x:x+w]
        return cropped_image
    else:
        # Если ненулевые пиксели не найдены, возвращаем исходное изображение
        return image_uint8

def resize_mask_to_contour(mask, target_contour):
    """
    Масштабирует маску до размеров ограничивающего прямоугольника целевого контура.
    :param mask: Маска (numpy array), которую нужно масштабировать.
    :param target_contour: Контур целевого объекта.
    :return: Масштабированная маска.
    """
    # Получаем ограничивающий прямоугольник для целевого контура
    x, y, contour_width, contour_height = cv2.boundingRect(target_contour)

    # Масштабируем маску
    resized_mask = cv2.resize(mask, (contour_width, contour_height), interpolation=cv2.INTER_AREA)

    # Бинаризация после масштабирования
    _, resized_mask = cv2.threshold(resized_mask, 127, 255, cv2.THRESH_BINARY)

    return resized_mask


def resize_and_center_mask(mask, target_shape):
    """
    Изменяет масштаб и центрирует маску относительно целевого изображения.
    :param mask: Маска (numpy array), которую нужно изменить.
    :param target_shape: Размер целевого изображения (высота, ширина).
    :return: Масштабированная и центрированная маска.
    """
    # Получаем размеры маски и целевого изображения
    mask_h, mask_w = mask.shape
    target_h, target_w = target_shape

    # Масштабируем маску до размера целевого изображения
    scale_x = target_w / mask_w
    scale_y = target_h / mask_h
    scale = min(scale_x, scale_y)  # Масштабируем, сохраняя пропорции
    new_w, new_h = int(mask_w * scale), int(mask_h * scale)

    resized_mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    # Создаем пустое изображение с размерами целевого изображения
    centered_mask = np.zeros((target_h, target_w), dtype=np.uint8)

    # Определяем смещения для центрирования
    offset_x = (target_w - new_w) // 2
    offset_y = (target_h - new_h) // 2

    # Помещаем масштабированную маску в центр нового изображения
    centered_mask[offset_y:offset_y+new_h, offset_x:offset_x+new_w] = resized_mask

    return centered_mask

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

# Функция для центрирования изображения
def center_image(image, binary_mask):
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
            h, w = binary_mask.shape
            shift_x, shift_y = w // 2 - cx, h // 2 - cy
            translation_matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
            centered_image = cv2.warpAffine(image, translation_matrix, (w, h))
            return centered_image
    return image

# Шаг 2: Поворот модели, рендеринг и обработка маски
min_difference = float('inf')
best_rotation = None
target_image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
_, target_binary = cv2.threshold(target_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# Поворот модели по всем возможным комбинациям поворотов осей X, Y и Z
for angle_x in range(0, 360, 120):  # Поворот по оси X
    for angle_y in range(0, 360, 120):  # Поворот по оси Y
        for angle_z in range(0, 360, 120):  # Поворот по оси Z
            # Преобразование углов в радианы
            angle_x_rad = np.radians(angle_x)
            angle_y_rad = np.radians(angle_y)
            angle_z_rad = np.radians(angle_z)

            # Создание матрицы поворота для каждой оси
            rotation_matrix_x = trimesh.transformations.rotation_matrix(angle_x_rad, [1, 0, 0], mesh.centroid)
            rotation_matrix_y = trimesh.transformations.rotation_matrix(angle_y_rad, [0, 1, 0], mesh.centroid)
            rotation_matrix_z = trimesh.transformations.rotation_matrix(angle_z_rad, [0, 0, 1], mesh.centroid)

            # Применяем матрицы поворота
            combined_rotation_matrix = np.dot(np.dot(rotation_matrix_x, rotation_matrix_y), rotation_matrix_z)
            rotated_mesh = mesh
            rotated_mesh.apply_transform(combined_rotation_matrix)

            # Создание сцены и рендеринг
            scene = trimesh.Scene(rotated_mesh)
            scene.camera.resolution = (512, 512)
            scene.camera.fov = (90, 90)

            # Проверка на размеры модели (если модель слишком маленькая)
            min_bound, max_bound = rotated_mesh.bounds
            if np.allclose(min_bound, max_bound):  # Проверка на нулевые размеры
                print(f"Ошибка: размеры модели слишком малы или нулевые для поворота ({angle_x}, {angle_y}, {angle_z})")
                continue

            # Находим центр модели
            center_point = (min_bound + max_bound) / 2
            scene.camera.look_at([center_point], distance=2)

            # Проверка, что камера не делит на ноль
            if np.any(np.isnan(center_point)) or np.isinf(center_point[0]) or np.isinf(center_point[1]) or np.isinf(center_point[2]):
                print(f"Ошибка: некорректная позиция камеры для поворота ({angle_x}, {angle_y}, {angle_z})")
                continue

            try:
                # Рендеринг изображения
                image_data = scene.save_image(background=[0, 0, 0, 255])
                image = Image.open(io.BytesIO(image_data))

                # Конвертация в оттенки серого и бинаризация
                gray_image = image.convert("L")
                binary_image = np.array(gray_image.point(lambda x: 255 if x > 1 else 0, mode='1'))

                # Обрезка маски по содержимому
                cropped_mask = crop_to_content(binary_image)

                # Поиск самого крупного контура в целевом изображении
                contours, _ = cv2.findContours(target_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)

                    # Масштабируем маску до размеров контура
                    resized_mask = resize_mask_to_contour(cropped_mask, largest_contour)

                    # Обрезаем объект в целевом изображении по контуру
                    cropped_target = crop_to_contour(target_binary, largest_contour)

                    # Обрезаем маску аналогично
                    mask_contour = cv2.findContours(resized_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0]
                    cropped_mask_resized = crop_to_contour(resized_mask, mask_contour)

                    # Сравнение обрезанных контуров
                    mask_contours, _ = cv2.findContours(cropped_mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    target_contours, _ = cv2.findContours(cropped_target, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    difference = 0
                    for c1, c2 in zip(mask_contours, target_contours):
                        difference += cv2.matchShapes(c1, c2, cv2.CONTOURS_MATCH_I1, 0.0)

                    # Проверяем, если текущий угол дает минимальную разницу
                    if difference < min_difference:
                        min_difference = difference
                        best_rotation = (angle_x, angle_y, angle_z)

                    print(f"Поворот: ({angle_x}, {angle_y}, {angle_z}) градусов, Разница: {difference}")
            except ZeroDivisionError as e:
                print(f"Ошибка деления на ноль при рендеринге ({angle_x}, {angle_y}, {angle_z}): {e}")
                continue
            except Exception as e:
                print(f"Ошибка рендеринга для поворота ({angle_x}, {angle_y}, {angle_z}): {e}")
                continue

print(f"Лучший угол поворота: {best_rotation}, Минимальная разница: {min_difference}")

# # Применяем лучший угол поворота к модели
angle_x, angle_y, angle_z = best_rotation

if angle_x >= 30: 
    angle_x = angle_x-30
if angle_y >= 30: 
    angle_y = angle_y-30    
if angle_z >= 30: 
    angle_z = angle_z-30
best_rotation, min_difference = rot(angle_x, angle_y, angle_z, 31, 20, mesh, min_difference, best_rotation, target_binary)


print(f"Лучший угол поворота: {best_rotation}, Минимальная разница: {min_difference}")

angle_x, angle_y, angle_z = best_rotation
if angle_x >= 10: 
    angle_x = angle_x-10
if angle_y >= 10: 
    angle_y = angle_y-10    
if angle_z >= 10: 
    angle_z = angle_z-10
best_rotation, min_difference = rot(angle_x, angle_y, angle_z, 11, 4, mesh, min_difference, best_rotation, target_binary)

print(f"Лучший угол поворота: {best_rotation}, Минимальная разница: {min_difference}")

angle_x, angle_y, angle_z = best_rotation
if angle_x >= 4: 
    angle_x = angle_x-4
if angle_y >= 4: 
    angle_y = angle_y-4   
if angle_z >= 4: 
    angle_z = angle_z-4
# Поворот модели по всем возможным комбинациям поворотов осей X, Y и Z
best_rotation, min_difference = rot(angle_x, angle_y, angle_z, 4, 2, mesh, min_difference, best_rotation, target_binary)
print(f"Лучший угол поворота: {best_rotation}, Минимальная разница: {min_difference}")

angle_x_rad = np.radians(angle_x)
angle_y_rad = np.radians(angle_y)
angle_z_rad = np.radians(angle_z)

rotation_matrix_x = trimesh.transformations.rotation_matrix(angle_x_rad, [1, 0, 0], mesh.centroid)
rotation_matrix_y = trimesh.transformations.rotation_matrix(angle_y_rad, [0, 1, 0], mesh.centroid)
rotation_matrix_z = trimesh.transformations.rotation_matrix(angle_z_rad, [0, 0, 1], mesh.centroid)

combined_rotation_matrix = np.dot(np.dot(rotation_matrix_x, rotation_matrix_y), rotation_matrix_z)
rotated_mesh = mesh
rotated_mesh.apply_transform(combined_rotation_matrix)

# Сохранение повернутой модели
output_obj_file_rotated = 'rotated_model.obj'
rotated_mesh.export(output_obj_file_rotated)
print(f"Повернутая модель сохранена как {output_obj_file_rotated}")
