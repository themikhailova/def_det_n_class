import trimesh
import numpy as np
import io
import copy
import time
import cv2
from compare import difference
from concurrent.futures import ProcessPoolExecutor
import pyrender
from PIL import Image, ImageOps


def rot(angle_x, angle_y, angle_z, mesh, invert_colors=True):
    '''
    Поворачивает модель на заданный угол с использованием pyrender.
    :param angle_x: угол поворота по оси X в градусах
    :param angle_y: угол поворота по оси Y в градусах
    :param angle_z: угол поворота по оси Z в градусах
    :param mesh: 3D-модель (объект trimesh)
    :param invert_colors: флаг для инверсии цветов изображения (по умолчанию False)
    :return: изображение модели, повернутой на заданный угол
    '''
    angle_x_rad = np.radians(angle_x)
    angle_y_rad = np.radians(angle_y)
    angle_z_rad = np.radians(angle_z)

    # Матрицы поворота
    rotation_matrix_x = trimesh.transformations.rotation_matrix(angle_x_rad, [1, 0, 0], mesh.centroid)
    rotation_matrix_y = trimesh.transformations.rotation_matrix(angle_y_rad, [0, 1, 0], mesh.centroid)
    rotation_matrix_z = trimesh.transformations.rotation_matrix(angle_z_rad, [0, 0, 1], mesh.centroid)

    # Комбинированная матрица поворота
    combined_rotation_matrix = np.dot(np.dot(rotation_matrix_x, rotation_matrix_y), rotation_matrix_z)

    # Применяем поворот к копии модели
    rotated_mesh = mesh.copy()
    rotated_mesh.apply_transform(combined_rotation_matrix)

    # --- Создание сцены с pyrender ---
    scene = pyrender.Scene()

    # Преобразование модели для использования с pyrender
    trimesh_mesh = pyrender.Mesh.from_trimesh(rotated_mesh, smooth=True)
    scene.add(trimesh_mesh)

    # Настройка источника света
    light = pyrender.PointLight(color=np.ones(3), intensity=50.0)  # Настраиваем интенсивность света
    light_pose = np.eye(4)  # Положение источника света (по умолчанию в центре)
    scene.add(light, pose=light_pose)

    max_extent, center, distance_mesh = calculate_model_scale_and_camera_distance(rotated_mesh)
    # print((3*max_extent)/2, distance_mesh)
    # Камера с фиксированным углом обзора
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 2.0)

    camera_pose = np.array([
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 2.5],
                [0.0, 0.0, 0.0, 1.0],
            ])
    scene.add(camera, pose=camera_pose)

    # --- Рендеринг сцены ---
    r = pyrender.OffscreenRenderer(1024, 768)  # Создаём рендерер для захвата изображения
    color, _ = r.render(scene)  # Рендерим сцену в изображение

    # Преобразование результата в изображение
    image = Image.fromarray(color)

    # Инверсия цветов, если флаг установлен
    if invert_colors:
        image = ImageOps.invert(image)

    # Очищаем рендерер
    r.delete()

    return image

def process_direction(args):
    angle, file_name, mesh, target_image, min_dif = args
    angle_x, angle_y, angle_z = angle

    image = rot(angle_x, angle_y, angle_z, mesh)

    if image is None:
        return None, min_dif
    
    image.save(file_name, "JPEG")
    model_image = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
    dif,_ = difference(model_image, target_image)

    return angle, dif if dif is not None and dif < min_dif else min_dif

def render_and_save_image_parallel(mesh, target_image, min_dif):
    directions = [
        ((0, 0, 0), './sides/front.jpg'),
        ((180, 0, 0), './sides/back.jpg'),
        ((90, 0, 0), './sides/top.jpg'),
        ((270, 0, 0), './sides/bottom.jpg'),
        ((0, 90, 0), './sides/right.jpg'),
        ((0, -90, 0), './sides/left.jpg')
    ]
    best_angles = (0, 0, 0)

    with ProcessPoolExecutor() as executor:
        results = executor.map(
            process_direction,
            [(angle, file_name, mesh, target_image, min_dif) for angle, file_name in directions]
        )
        for angle, dif in results:
            print(angle, dif)
            if dif < min_dif:
                min_dif = dif
                best_angles = angle

    return best_angles, min_dif

def calculate_model_scale_and_camera_distance(mesh, yfov=np.pi / 3.0):
    """
    Рассчитывает масштаб модели и минимальное расстояние камеры для полного обзора.
    :param mesh: 3D-модель (trimesh object)
    :param yfov: Поле зрения камеры (в радианах)
    :return: масштаб модели, минимальное расстояние камеры
    """
    # Получаем размеры модели
    bounding_box = mesh.bounding_box.extents  # Длина, ширина, высота
    max_extent = max(bounding_box)  # Максимальный размер вдоль осей
    center = mesh.bounding_box.centroid  # Центр модели

    # Минимальное расстояние камеры для полного обзора модели
    distance = max_extent

    return max_extent, center, distance

def move_model(x, y, z, mesh, invert_colors=True):
    '''
    Перемещает модель в заданные координаты относительно сцены, фиксируя камеру и настраивая освещение как в функции rot.
    :param x, y, z: координаты смещения модели
    :param mesh: 3D-модель
    :return: изображение модели с новой позиции
    '''
    # Перемещаем модель
    translation_matrix = trimesh.transformations.translation_matrix([x, y, z])
    moved_mesh = mesh.copy()
    moved_mesh.apply_transform(translation_matrix)

    # --- Создание сцены с pyrender ---
    scene = pyrender.Scene()

    # Преобразование модели для использования с pyrender
    pyrender_mesh = pyrender.Mesh.from_trimesh(moved_mesh, smooth=True)
    scene.add(pyrender_mesh)
    if x == 0 and y == 0:
        light = pyrender.PointLight(color=np.ones(3), intensity=5.0)
    else:
        # --- Настройка освещения (как в rot) ---
        light = pyrender.PointLight(color=np.ones(3), intensity=50.0)  # Интенсивность света
    light_pose = np.eye(4)  # Свет в фиксированной позиции (в центре сцены)
    scene.add(light, pose=light_pose)

    # Рассчитываем масштаб и расстояние камеры
    max_extent, center, distance_mesh = calculate_model_scale_and_camera_distance(moved_mesh)
    # print((3*max_extent)/2, distance_mesh)
    # Камера с фиксированным углом обзора
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 2.0)

    camera_pose = np.array([
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, (3*max_extent+0.4)/2],
                [0.0, 0.0, 0.0, 1.0],
            ])
  
    scene.add(camera, pose=camera_pose)

    # --- Рендеринг сцены ---
    r = pyrender.OffscreenRenderer(1024, 1024)  # Создаем рендерер для захвата изображения
    color, _ = r.render(scene)  # Рендерим сцену в изображение

    # Преобразование результата в изображение
    image = Image.fromarray(color)
    if invert_colors:
        image = ImageOps.invert(image)
    # Очищаем рендерер
    r.delete()

    return image

def refine_search(mesh, target_image, min_dif, initial_step=1.0, min_step=0.1, step_factor=0.5, points_to_check=5, threshold_points_to_stop=3):
    """
    Реализация итеративного поиска минимальной точки с досрочным завершением итерации.
    
    :param mesh: 3D-модель
    :param target_image: Целевое изображение
    :param min_dif: Текущая минимальная разница
    :param initial_step: Начальный шаг для поиска
    :param min_step: Минимальный шаг для остановки поиска
    :param step_factor: Фактор уменьшения шага на каждой итерации
    :param points_to_check: Число точек для проверки в первой итерации
    :param threshold_points_to_stop: Число последовательных увеличений разницы для прекращения итерации
    :return: Координаты минимальной точки и минимальная разница
    """
    def generate_points_around(center, step):
        """Генерация точек вокруг заданного центра с указанным шагом."""
        x, y, z = center
        points = []
        for dx in np.arange(-step, step + step / points_to_check, step / points_to_check):
            for dy in np.arange(-step, step + step / points_to_check, step / points_to_check):
                points.append((x + dx, y + dy, z))
        return points
    
    def generate_points_in_circle(center, step, points_to_check):
        """Генерация точек по кругу вокруг заданного центра."""
        x, y, z = center
        angles = np.linspace(0, 2 * np.pi, points_to_check, endpoint=False)  # Углы круга
        points = []
        for angle in angles:
            dx = step * np.cos(angle)
            dy = step * np.sin(angle)
            points.append((x + dx, y + dy, z))  # Точки на окружности
        return points


    def evaluate_points(points):
        """Оценка разницы для списка точек."""
        results = []
        for point in points:
            image = move_model(*point, mesh)
            if image is None:
                continue
            model_image = np.array(image.convert('L'))  # Преобразование в grayscale
            dif, _ = difference(model_image, target_image)
            if dif is not None:  # Проверяем, что разница не None
                results.append((point, dif))
        return results

    max_extent, center, distance_mesh = calculate_model_scale_and_camera_distance(mesh)
    current_step = max_extent
    current_point = center  # Стартовая точка
    best_point = current_point

    while current_step > min_step:
        print(current_step)
        points = generate_points_in_circle(current_point, current_step, points_to_check)
        # points = generate_points_around(current_point, current_step)
        results = []
        consecutive_increases = 0  # Счетчик увеличений разницы
        prev_dif = float('inf')  # Начальное значение для сравнения разницы
        
        for point in points:
            # Оцениваем текущую точку
            print(f"Текущая точка: x={point[0]:.3f}, y={point[1]:.3f}, z={point[2]:.3f}")
            image = move_model(*point, mesh)
            if image is None:
                continue
            model_image = np.array(image.convert('L'))
            dif, _ = difference(model_image, target_image)
            if dif is not None:
                results.append((point, dif))
                
                # Проверка последовательного увеличения разницы
                if dif >= prev_dif:
                    consecutive_increases += 1
                else:
                    consecutive_increases = 0  # Сбрасываем счетчик при улучшении
                
                prev_dif = dif
                print('dif: ', dif, consecutive_increases)
                # Если превышен порог увеличений, прекращаем текущую итерацию
                if consecutive_increases >= threshold_points_to_stop:
                    break

        # Если результаты пустые, уменьшаем шаг и продолжаем
        if not results:
            current_step *= step_factor
            continue

        # Сортируем точки по разнице
        results.sort(key=lambda x: x[1])

        # Берем точку с минимальной разницей
        best_candidate, best_dif = results[0]

        if best_dif < min_dif:
            min_dif = best_dif
            best_point = best_candidate
            current_point = best_candidate
        else:
            # Уменьшаем шаг, так как дальнейшее улучшение не найдено
            current_step *= step_factor

    return best_point, min_dif



if __name__ == '__main__':
    
    input_img = r'./fig2.jpg'
    input_obj_file = r'./mod2.obj'

    mesh_or = trimesh.load(input_obj_file)
    min_dif = float('inf')
    target_image = cv2.imread(input_img, cv2.IMREAD_GRAYSCALE)

    start = time.time()

    angles, min_dif = render_and_save_image_parallel(mesh_or, target_image, min_dif)
    print(angles, min_dif)

    angle_x, angle_y, angle_z = angles
    angle_x_rad = np.radians(angle_x)
    angle_y_rad = np.radians(angle_y)
    angle_z_rad = np.radians(angle_z)
    rotation_matrix_x = trimesh.transformations.rotation_matrix(angle_x_rad, [1, 0, 0], mesh_or.centroid)
    rotation_matrix_y = trimesh.transformations.rotation_matrix(angle_y_rad, [0, 1, 0], mesh_or.centroid)
    rotation_matrix_z = trimesh.transformations.rotation_matrix(angle_z_rad, [0, 0, 1], mesh_or.centroid)
    combined_rotation_matrix = np.dot(np.dot(rotation_matrix_x, rotation_matrix_y), rotation_matrix_z)
    rotated_mesh = mesh_or.copy()
    rotated_mesh.apply_transform(combined_rotation_matrix)
    # best_camera_angles, min_difference = render_images_camera_rotation(rotated_mesh, target_image, min_dif)

    best_camera_angles, min_difference = refine_search(rotated_mesh, target_image, min_dif)


    print(best_camera_angles, min_difference, angles, min_dif)

    scene = trimesh.Scene(rotated_mesh)
    scene.camera.resolution = (1024, 1024)
    x, y, z = best_camera_angles
    cam_rot = np.array([0, 0, 0])
    center_point = np.array(rotated_mesh.centroid) + np.array([x, y, z]) 
    print(center_point)
    scene.set_camera(angles=cam_rot, distance=60, center=center_point)

    output_obj_file_rotated = './rotated_model.obj'
    rotated_mesh.export(output_obj_file_rotated)
    finish = time.time()
    print(f"Повернутая модель: {output_obj_file_rotated}")

    rotated_mesh.show()

    res_msec = (finish - start) * 1000
    print('Время работы в миллисекундах: ', res_msec)

    print("Изображения успешно сохранены.")
