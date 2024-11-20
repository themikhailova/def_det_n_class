import trimesh
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import io

input_obj_file = 'details_v1.obj'
output_mask_file = 'mask.jpg'
input_image_path = 'det_orig.jpg'

mesh = trimesh.load(input_obj_file)


# обрезка маски
def crop_to_content(image):
    image_uint8 = (image * 255).astype(np.uint8)  
    coords = cv2.findNonZero(image_uint8)  # ненулевые пиксели
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)  # ограничивающий прямоугольник
        cropped_image = image_uint8[y:y+h, x:x+w]
        return cropped_image
    else:
        # если ненулевые пиксели не найдены, то исходное изображение
        return image_uint8

def resize_mask_to_contour(mask, target_contour):
    """
    Масштабирует маску до размеров ограничивающего прямоугольника целевого контура
    :param mask: Маска (numpy array), которую нужно масштабировать
    :param target_contour: Контур целевого объекта
    :return: Масштабированная маска
    """
    x, y, contour_width, contour_height = cv2.boundingRect(target_contour)
    resized_mask = cv2.resize(mask, (contour_width, contour_height), interpolation=cv2.INTER_AREA)

    _, resized_mask = cv2.threshold(resized_mask, 127, 255, cv2.THRESH_BINARY)

    return resized_mask


def resize_and_center_mask(mask, target_shape):
    """
    Изменяет масштаб и центрирует маску относительно целевого изображения
    :param mask: Маска (numpy array), которую нужно изменить
    :param target_shape: Размер целевого изображения (высота, ширина)
    :return: Масштабированная и центрированная маска
    """
    mask_h, mask_w = mask.shape
    target_h, target_w = target_shape

    # масштабируем маску до размера целевого изображения
    scale_x = target_w / mask_w
    scale_y = target_h / mask_h
    scale = min(scale_x, scale_y)  #  сохраняем пропорции
    new_w, new_h = int(mask_w * scale), int(mask_h * scale)

    resized_mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    centered_mask = np.zeros((target_h, target_w), dtype=np.uint8)

    # смещения для центрирования
    offset_x = (target_w - new_w) // 2
    offset_y = (target_h - new_h) // 2

    centered_mask[offset_y:offset_y+new_h, offset_x:offset_x+new_w] = resized_mask

    return centered_mask

def crop_to_contour(image, contour):
    """
    Обрезает изображение по ограничивающему прямоугольнику заданного контура
    :param image: Исходное изображение (numpy)
    :param contour: Контур объекта
    :return: Обрезанное изображение
    """
    x, y, contour_width, contour_height = cv2.boundingRect(contour)
    cropped_image = image[y:y+contour_height, x:x+contour_width]
    return cropped_image

def align_angle(image, output_path=None, show_result=False):
    """
    выравнивание детали на изображении, учитывая небольшой наклон
    
    :param image_path: Путь к входному изображению
    :param output_path: Путь для сохранения выровненного изображения
    :param show_result: Флаг для отображения результата
    :return: Выровненное изображение
    """
    
    # оттенки серого
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # размытие для удаления шума
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # выделение контуров
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        raise ValueError("Контуры не найдены")
    
    largest_contour = max(contours, key=cv2.contourArea)
    
    rect = cv2.minAreaRect(largest_contour)
    angle = rect[-1]
    print(angle)
    # Корректировка угла
    # if angle < -45:
    #     angle += 90
    
    # центр изображения для поворота
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    
    # поворот изображения
    rotation_matrix = cv2.getRotationMatrix2D(center, angle-90, 1.0)
    rotated = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    if output_path:
        cv2.imwrite(output_path, rotated)
    
    if show_result:
        cv2.imshow("Aligned Image", rotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return rotated

directions = [
    (0, 0, 0),       # фронт
    (180, 0, 0),     # задняя сторона
    (90, 0, 0),      # слева
    (270, 0, 0),     # справа
    (0, 90, 0),      # сверху
    (0, -90, 0)      # снизу
]

min_difference = float('inf')
best_rotation = None

target_image = cv2.imread(input_image_path, cv2.IMREAD_ANYCOLOR)
target_image_rot = align_angle(target_image, './rot_det.jpg', True)
target_image = cv2.imread('./rot_det.jpg', cv2.IMREAD_GRAYSCALE)

_, target_binary = cv2.threshold(target_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
contours, _ = cv2.findContours(target_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for angle_x, angle_y, angle_z in directions:
    try:
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

        scene = trimesh.Scene(rotated_mesh)
        scene.camera.resolution = (512, 512)
        scene.camera.fov = (90, 90)
        min_bound, max_bound = rotated_mesh.bounds
        center_point = (min_bound + max_bound) / 2
        scene.camera.look_at([center_point], distance=2)

        image_data = scene.save_image(background=[0, 0, 0, 255])
        image = Image.open(io.BytesIO(image_data))
        
        if contours:
            # оттенки серого и бинаризация
            gray_image = image.convert("L")
            binary_image = np.array(gray_image.point(lambda x: 255 if x > 1 else 0, mode='1'))
            cropped_mask = crop_to_content(binary_image)
            largest_contour = max(contours, key=cv2.contourArea)
            # масштабируем маску до размеров контура
            resized_mask = resize_mask_to_contour(cropped_mask, largest_contour)

            # обрезаем объект в целевом изображении по контуру
            cropped_target = crop_to_contour(target_binary, largest_contour)

            # обрезаем маску аналогично
            mask_contour = cv2.findContours(resized_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0]
            cropped_mask_resized = crop_to_contour(resized_mask, mask_contour)

            # сравнение обрезанных контуров
            mask_contours, _ = cv2.findContours(cropped_mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            target_contours, _ = cv2.findContours(cropped_target, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            difference = 0
            for c1, c2 in zip(mask_contours, target_contours):
                difference += cv2.matchShapes(c1, c2, cv2.CONTOURS_MATCH_I1, 0.0)

            if difference < min_difference:
                min_difference = difference
                best_rotation = (angle_x, angle_y, angle_z)

            print(f"Поворот: ({angle_x}, {angle_y}, {angle_z}) градусов, Разница: {difference}")

    except Exception as e:
        print(f"Ошибка при обработке углов ({angle_x}, {angle_y}, {angle_z}): {e}")

print(f"Лучший угол поворота из 6 сторон: {best_rotation}, Минимальная разница: {min_difference}")

angle_x, angle_y, angle_z = best_rotation

angle_x_rad = np.radians(angle_x)
angle_y_rad = np.radians(angle_y)
angle_z_rad = np.radians(angle_z)

rotation_matrix_x = trimesh.transformations.rotation_matrix(angle_x_rad, [1, 0, 0], mesh.centroid)
rotation_matrix_y = trimesh.transformations.rotation_matrix(angle_y_rad, [0, 1, 0], mesh.centroid)
rotation_matrix_z = trimesh.transformations.rotation_matrix(angle_z_rad, [0, 0, 1], mesh.centroid)

combined_rotation_matrix = np.dot(np.dot(rotation_matrix_x, rotation_matrix_y), rotation_matrix_z)
rotated_mesh = mesh.copy()
rotated_mesh.apply_transform(combined_rotation_matrix)

output_obj_file_rotated = 'rotated_model_v1.obj'
rotated_mesh.export(output_obj_file_rotated)
print(f"Повернутая модель сохранена как {output_obj_file_rotated}")

rotated_mesh.show() 
