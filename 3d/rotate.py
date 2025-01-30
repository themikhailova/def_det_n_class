import trimesh
import numpy as np
import sys
import cv2
from compare import difference
from concurrent.futures import ProcessPoolExecutor
import pyrender
from PIL import Image, ImageOps
import os
import logging

sys.path.append(os.path.abspath('./threed'))
from threed.det_aligning import align
sys.path.append(os.path.abspath('./detect'))
from detect.character import remover

logging.basicConfig(level=logging.INFO)

RENDER_WIDTH = 1024
RENDER_HEIGHT = 768
LIGHT_INTENSITY = 55.0
LIGHT_INTENSITY_HIGH = 75.0
CAMERA_FOV = np.pi / 2.0
DISTANCE_SCALE_FACTOR = 3.0
DISTANCE_OFFSET = 0.4


def check_file_exists(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Файл не найден: {file_path}")

def rot(angle_x, angle_y, angle_z, mesh, invert_colors=True):
    '''
    Поворачивает 3D-модель на заданные углы вокруг осей X, Y и Z и рендерит изображение модели

    Используется на первом этапе для проверки шести сторон модели

    Args:
        angle_x (float): Угол поворота по оси X в градусах
        angle_y (float): Угол поворота по оси Y в градусах
        angle_z (float): Угол поворота по оси Z в градусах
        mesh (trimesh.Trimesh): 3D-модель, которую нужно повернуть
        invert_colors (bool, optional): Флаг для инверсии цветов изображения По умолчанию True

    Returns:
        PIL.Image.Image: Изображение модели, повернутой на заданные углы

    Raises:
        RuntimeError: Если в результате рендеринга получено пустое изображение
    '''
    if not isinstance(angle_x, (int, float)):
        raise TypeError(f"angle_x должен быть числом, получено {type(angle_x)}")

    if not isinstance(angle_y, (int, float)):
        raise TypeError(f"angle_y должен быть числом, получено {type(angle_y)}")

    if not isinstance(angle_z, (int, float)):
        raise TypeError(f"angle_z должен быть числом, получено {type(angle_z)}")

    if not isinstance(mesh, (trimesh.Trimesh, trimesh.Scene)):
        raise TypeError(f"mesh должен быть экземпляром trimesh.Trimesh или trimesh.Scene, получено {type(mesh)}")

    if not isinstance(invert_colors, bool):
        raise TypeError(f"invert_colors должен быть bool, получено {type(invert_colors)}")
    if isinstance(mesh, trimesh.Trimesh) and len(mesh.vertices) == 0:
        raise ValueError("Переданный mesh пуст (не содержит вершин)")

    if isinstance(mesh, trimesh.Scene) and len(mesh.geometry) == 0:
        raise ValueError("Переданная сцена (Scene) не содержит объектов")

    logging.info(f"Запуск функции rot для углов: X={angle_x}, Y={angle_y}, Z={angle_z}")
    if isinstance(mesh, trimesh.Scene):
        dumped_meshes = mesh.dump()
        if not dumped_meshes:
            raise ValueError("Не удалось извлечь геометрию из сцены")
        mesh = trimesh.util.concatenate(dumped_meshes)

    # Преобразуем углы из градусов в радианы
    angle_x_rad = np.radians(angle_x)
    angle_y_rad = np.radians(angle_y)
    angle_z_rad = np.radians(angle_z)

    # Создаем матрицы поворота для каждой оси
    rotation_matrix_x = trimesh.transformations.rotation_matrix(angle_x_rad, [1, 0, 0], mesh.centroid)
    rotation_matrix_y = trimesh.transformations.rotation_matrix(angle_y_rad, [0, 1, 0], mesh.centroid)
    rotation_matrix_z = trimesh.transformations.rotation_matrix(angle_z_rad, [0, 0, 1], mesh.centroid)

    # Комбинированная матрица поворота
    combined_rotation_matrix = np.dot(np.dot(rotation_matrix_x, rotation_matrix_y), rotation_matrix_z)

    # Применяем поворот к копии модели
    rotated_mesh = mesh.copy()
    rotated_mesh.apply_transform(combined_rotation_matrix)

    # Создаем сцену для рендеринга
    scene = pyrender.Scene()
    trimesh_mesh = pyrender.Mesh.from_trimesh(rotated_mesh, smooth=True)
    scene.add(trimesh_mesh)

    # Настройка источника света
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=LIGHT_INTENSITY)
    light_pose = np.array([
        [1.0, 0.0, 0.0, 2.0],
        [0.0, 1.0, 0.0, 2.0],
        [0.0, 0.0, 1.0, .0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    scene.add(light, pose=light_pose)

    # Камера с фиксированным углом обзора
    camera = pyrender.PerspectiveCamera(yfov=CAMERA_FOV)
    camera_pose = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 2.5],
        [0.0, 0.0, 0.0, 1.0],
    ])
    scene.add(camera, pose=camera_pose)

    # Рендеринг сцены
    try:
        r = pyrender.OffscreenRenderer(RENDER_WIDTH, RENDER_HEIGHT)
        color, _ = r.render(scene)
    except Exception as e:
        raise RuntimeError(f"Ошибка рендеринга сцены: {e}")

    if color is None or color.size == 0:
        raise RuntimeError("Ошибка рендеринга: получено пустое изображение")

    # Преобразование в изображение
    image = Image.fromarray(color)
    if image is None:
        raise RuntimeError("Ошибка создания изображения из массива")


    # Инверсия цветов, если флаг установлен
    if invert_colors:
        image = ImageOps.invert(image)
    image = ImageOps.autocontrast(image)
    # Очищаем рендерер
    if r:
        r.delete()

    # Проверка на пустое изображение
    if image is None or image.size == 0:
        raise RuntimeError("Ошибка рендеринга: получено пустое изображение")

    logging.info(f"Функция rot успешно завершена для углов: X={angle_x}, Y={angle_y}, Z={angle_z}")
    return image

def process_direction(args):
    '''
    Обрабатывает заданный угол поворота модели, рендерит её и вычисляет разницу между результатом и целевым изображением

    Args:
        args (tuple): Кортеж, содержащий:
            - angle (tuple): Углы поворота модели (X, Y, Z) в градусах
            - file_name (str): Имя файла для сохранения рендеринга
            - mesh (trimesh.Trimesh): 3D-модель для обработки
            - target_image (numpy.ndarray): Целевое изображение в формате grayscale
            - min_dif (float): Текущее минимальное значение разницы

    Returns:
        tuple:
            - angle (tuple): Углы поворота модели (X, Y, Z), если обработка успешна
            - difference (float): Разница между рендерингом и целевым изображением,
              или текущее минимальное значение разницы, если вычисленная больше

    Raises:
        Exception: Если в процессе выполнения возникает ошибка
    '''
    if not isinstance(args, tuple) or len(args) != 5:
        raise TypeError(f"args должен быть кортежем из 5 элементов, получено {type(args)} с длиной {len(args) if isinstance(args, tuple) else 'N/A'}")

    try:
        # Распаковка аргументов
        angle, file_name, mesh, target_image, min_dif = args
        if not isinstance(angle, tuple) or len(angle) != 3:
            raise TypeError(f"angle должен быть кортежем из 3 чисел, получено {type(angle)}")

        angle_x, angle_y, angle_z = angle
        if not all(isinstance(a, (int, float)) for a in angle):
            raise TypeError(f"Элементы angle должны быть числами, получено {angle}")

        # Рендеринг изображения с заданными углами
        try:
            image = rot(angle_x, angle_y, angle_z, mesh)
            if image is None:
                raise RuntimeError(f"Не удалось сгенерировать изображение для углов: {angle}")
        except Exception as e:
            raise RuntimeError(f"Ошибка при рендеринге модели: {e}")

        if image is None:
            logging.warning(f"Не удалось сгенерировать изображение для углов: X={angle_x}, Y={angle_y}, Z={angle_z}")
            return None, min_dif

        # Сохранение рендеринга в файл
        image.save(file_name, "JPEG")

        # Загрузка сохраненного изображения в grayscale
        if not os.path.exists(file_name):
            raise FileNotFoundError(f"Файл {file_name} не найден")

        model_image = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
        if model_image is None:
            raise RuntimeError(f"Ошибка загрузки изображения {file_name}")

        if model_image is None:
            logging.error(f"Не удалось загрузить изображение из файла: {file_name}")
            return None, min_dif

        # Вычисление разницы между рендерингом и целевым изображением
        dif, _ = difference(model_image, target_image)
        if dif is None:
            return None, min_dif

        # Логирование результата вычисления разницы
        logging.info(f"Разница для углов X={angle_x}, Y={angle_y}, Z={angle_z}: {dif}")

    except Exception as e:
        return None, float('inf')  # Возвращаем значение, чтобы не нарушить процесс работы

    # Возврат минимальной разницы или текущей разницы
    return angle, dif if dif < min_dif else min_dif

def render_and_save_image_parallel(mesh, target_image, min_dif):
    '''
    Параллельно обрабатывает набор углов поворота модели и выбирает углы с минимальной разницей

    Args:
        mesh (trimesh.Trimesh): 3D-модель для обработки
        target_image (numpy.ndarray): Целевое изображение в формате grayscale
        min_dif (float): Текущее минимальное значение разницы

    Returns:
        tuple:
            - best_angles (tuple): Углы поворота (X, Y, Z), которые дают минимальную разницу
            - min_dif (float): Минимальная разница между рендерингом и целевым изображением
    '''
    
    logging.info("Запуск функции render_and_save_image_parallel для поиска оптимальных углов")

    # Задаем направления (углы поворота) и файлы для сохранения рендеров
    directions = [
        ((0, 0, 0), './sides/front.jpg'),
        ((180, 0, 0), './sides/back.jpg'),
        ((90, 0, 0), './sides/top.jpg'),
        ((270, 0, 0), './sides/bottom.jpg'),
        ((0, 90, 0), './sides/right.jpg'),
        ((0, -90, 0), './sides/left.jpg')
    ]

    best_angles = (0, 0, 0)  # Инициализация лучших углов поворота

    # Запускаем обработку углов последовательно
    try:        
        # Формируем аргументы для функции process_direction
        tasks = [(angle, file_name, mesh, target_image, min_dif) for angle, file_name in directions]
        
        for task in tasks:
            # Вызываем функцию process_direction для каждого задания
            angle, dif = process_direction(task)
            
            if angle is None or dif is None:
                continue
            
            logging.debug(f"Обработан угол: {angle}, разница: {dif}")
            
            # Проверяем, является ли текущий результат лучшим
            if dif < min_dif:
                min_dif = dif
                best_angles = angle
                logging.info(f"Найдены новые лучшие углы: {best_angles} с разницей: {min_dif}")
    except Exception as e:
        logging.error(f"Ошибка во время обработки: {str(e)}")
        raise

    logging.info("Завершение функции render_and_save_image_parallel")
    return best_angles, min_dif

def calculate_model_scale_and_camera_distance(mesh, yfov=CAMERA_FOV, render_width=RENDER_WIDTH, render_height=RENDER_HEIGHT, safety_margin=1.0, offset=(0, 0, 0)):
    '''
    Рассчитывает масштаб модели, её центр и минимальное расстояние камеры для полного обзора.

    Args:
        mesh (trimesh.Trimesh): 3D-модель для обработки.
        yfov (float, optional): Вертикальное поле зрения камеры (в радианах). По умолчанию π/3.
        render_width (int, optional): Ширина рендера в пикселях. По умолчанию 800.
        render_height (int, optional): Высота рендера в пикселях. По умолчанию 600.
        safety_margin (float, optional): Запас для уверенности, что модель попадёт в кадр. По умолчанию 1.1.

    Returns:
        tuple:
            - diagonal (float): Диагональ модели (более точная метрика её размера).
            - center (numpy.ndarray): Центр модели (координаты в 3D-пространстве).
            - distance (float): Минимальное расстояние камеры для полного обзора модели.
    '''
    logging.info("Запуск функции calculate_model_scale_and_camera_distance для 3D-модели")

    try:
        # Извлечение ориентированного ограничивающего параллелепипеда (OBB) модели
        bounding_box = mesh.bounding_box_oriented.extents  # Длина, ширина, высота модели в её ориентации

        # Определение максимального размера среди осей (максимальная длина модели)
        max_extent = max(bounding_box)
        # Вычисление диагонали ограничивающего параллелепипеда
        diagonal = np.linalg.norm(bounding_box)  # Диагональ 3D-модели

        # Вычисление центра модели (центроид модели)
        center = mesh.bounding_box.centroid
        # center[0] += offset[0]
        # center[1] += offset[1]
        # center[2] += offset[2]

        aspect_ratio = render_width / render_height
        diagonal *= aspect_ratio 
       
        # Расчёт минимального расстояния камеры с использованием диагонали
        distance = diagonal / (2 * np.tan(yfov / 2)) * safety_margin

        # Динамическая проверка попадания модели в кадр
        if not is_model_fully_in_frame(mesh, distance, yfov, render_width, render_height, offset):
            distance *= 1.5  # Увеличиваем расстояние камеры на 150%

        # Завершение функции с успешным результатом
        logging.info("Функция calculate_model_scale_and_camera_distance завершена успешно")
        return diagonal, center, distance

    except Exception as e:
        # Логирование ошибки в случае исключения
        logging.error(f"Ошибка в функции calculate_model_scale_and_camera_distance: {e}")
        raise

def is_model_fully_in_frame(mesh, camera_distance, yfov, render_width, render_height, offset):
    '''
    Проверяет, находится ли хотя бы часть модели в кадре, если она не полностью помещается.

    Args:
        mesh (trimesh.Trimesh): 3D модель для проверки.
        camera_distance (float): Расстояние от камеры до модели.
        yfov (float): Вертикальное поле зрения камеры (в радианах).
        render_width (int): Ширина рендера в пикселях.
        render_height (int): Высота рендера в пикселях.

    Returns:
        bool: True, если хотя бы часть модели находится в кадре.
    '''
    # Получаем размеры ограничивающего параллелепипеда модели
    bounding_box = mesh.bounding_box.extents
    max_x = bounding_box[0] / 2  # половина по оси X
    max_y = bounding_box[1] / 2  # половина по оси Y
    max_z = bounding_box[2] / 2  # половина по оси Z

    # Рассчитываем размер модели в пикселях
    projected_height = 2 * camera_distance * np.tan(yfov / 2)  # Высота в пикселях
    projected_width = projected_height * (render_width / render_height)  # Ширина в пикселях

    # Центр модели в 3D пространстве
    if not isinstance(offset, np.ndarray):
        offset = np.array(offset)

    center = mesh.bounding_box.centroid + offset
    model_x, model_y, model_z = center

    # Проекция модели на экран по осям X и Y
    x_min = model_x - max_x
    x_max = model_x + max_x
    y_min = model_y - max_y
    y_max = model_y + max_y

    # Проверка, если хотя бы одна часть модели выходит за пределы окна
    if (x_max*1.3 < -projected_width / 2 or x_min*1.3 > projected_width / 2 or
        y_max*1.3 < -projected_height / 2 or y_min*1.3 > projected_height / 2):
        return False  # Модель выходит хотя бы частично за пределы кадра

    # Если модель не выходит за пределы кадра
    return True

def move_model(x, y, z, mesh, invert_colors=True, save=False):
    '''
    Перемещает 3D-модель на заданное смещение в пространстве, рендерит её изображение 
    и предоставляет возможность сохранить изображение

    Args:
        x (float): Смещение модели по оси X
        y (float): Смещение модели по оси Y
        z (float): Смещение модели по оси Z
        mesh (trimesh.Trimesh): 3D-модель для обработки
        invert_colors (bool, optional): Флаг для инверсии цветов в изображении. По умолчанию True
        save (bool, optional): Флаг для сохранения изображения в высоком разрешении. По умолчанию False

    Returns:
        PIL.Image.Image: Сгенерированное изображение модели после перемещения
    '''
    logging.info(f"Запуск функции move_model для смещений x={x}, y={y}, z={z}")

    # Создаем матрицу трансляции для смещения модели и применяем ее к копии модели
    translation_matrix = trimesh.transformations.translation_matrix([x, y, z])
    if isinstance(mesh, trimesh.Scene):
        # Извлекаем геометрии из сцены и объединяем их в один объект Trimesh
        mesh = trimesh.util.concatenate(mesh.dump())
    moved_mesh = mesh.copy()
    moved_mesh.apply_transform(translation_matrix)

    # Создаем сцену
    scene = pyrender.Scene()
    
    # Преобразование модели для использования с pyrender и добавление модели на сцену
    pyrender_mesh = pyrender.Mesh.from_trimesh(moved_mesh, smooth=True)
    scene.add(pyrender_mesh)

    # Рассчитываем масштаб и расстояние камеры
    _, center, distance_mesh = calculate_model_scale_and_camera_distance(moved_mesh, offset=(x,y,z))
    center_x, center_y, _ = center
    # if save: 
    #     distance_mesh = 2.5 # для попадания в кадр крайних нужно минимум 2.5 но это слишком далеко для хорошего качества

    # Настройка освещения
    if (center_x == 0 and center_y == 0):
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=LIGHT_INTENSITY)
    else:
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=25.0) 

    light_pose = np.array([
        [1.0, 0.0, 0.0, 2.0],
        [0.0, 1.0, 0.0, 2.0],
        [0.0, 0.0, 1.0, 2.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
   
    scene.add(light, pose=light_pose)

    # Камера с фиксированным углом обзора
    camera = pyrender.PerspectiveCamera(yfov=CAMERA_FOV)
    camera_pose = np.array([
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 2.5],
                [0.0, 0.0, 0.0, 1.0],
            ])
    scene.add(camera, pose=camera_pose)

    # Рендеринг сцены
    r = pyrender.OffscreenRenderer(RENDER_WIDTH, RENDER_HEIGHT)  # Создаем рендерер для захвата изображения
    if save: 
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=35.0) 
        light_pose = np.array([
            [1.0, 0.0, 0.0, 2.0],
            [0.0, 1.0, 0.0, 2.0],
            [0.0, 0.0, 1.0, 2.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        scene.add(light, pose=light_pose)

        pyrender_mesh = pyrender.Mesh.from_trimesh(moved_mesh, smooth=True)
        scene.add(pyrender_mesh)

        camera = pyrender.PerspectiveCamera(yfov=CAMERA_FOV)
        camera_pose = np.array([
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 2.0],
                [0.0, 0.0, 0.0, 1.0],
            ])
        scene.add(camera, pose=camera_pose)

        color, _ = r.render(scene)
        image = Image.fromarray(color)
        if invert_colors:
            image = ImageOps.invert(image)
        image = ImageOps.autocontrast(image)
        r.delete()
        
    else:
        color, _ = r.render(scene)  # Рендерим сцену в изображение

        # Преобразование результата в изображение
        image = Image.fromarray(color)
        if invert_colors:
            image = ImageOps.invert(image)
        image = ImageOps.autocontrast(image)
        # Очищаем рендерер
        r.delete()

    if image is None or image.size == 0:
        raise RuntimeError("Ошибка рендеринга: получено пустое изображение")
    
    logging.info(f"Функция move_model успешно завершена для x={x}, y={y}, z={z}")
    
    # Визуализируем изображение
    # image.show()
    return image


def refine_search(mesh, target_image, min_dif, min_step=0.1, step_factor=0.5, points_to_check=5, threshold_points_to_stop=3):
    '''
    Реализует итеративный поиск минимальной точки для модели с досрочным завершением итерации

    Args:
        mesh (trimesh.Trimesh): 3D-модель для поиска
        target_image (numpy.ndarray): Целевое изображение для сравнения
        min_dif (float): Текущее минимальное значение разницы
        min_step (float, optional): Минимальный шаг, при котором поиск завершится. По умолчанию 0.1
        step_factor (float, optional): Фактор уменьшения шага на каждой итерации. По умолчанию 0.5
        points_to_check (int, optional): Число точек для проверки в каждой итерации. По умолчанию 5
        threshold_points_to_stop (int, optional): Количество последовательных увеличений разницы для завершения поиска. По умолчанию 3

    Returns:
        tuple:
            - best_point (tuple): Координаты точки с минимальной разницей (x, y, z)
            - min_dif (float): Минимальная разница между изображением модели и целевым изображением
    '''
    logging.info("Запуск функции refine_search для нахождения минимальной точки")
    
    def generate_points_in_circle(center, step, points_to_check):
        '''
        Генерация точек по кругу вокруг заданного центра с указанным шагом

        Args:
            center (tuple): Координаты центра круга (x, y, z)
            step (float): Шаг, который определяет радиус круга
            points_to_check (int): Количество точек, которые нужно генерировать на окружности

        Returns:
            list: Список сгенерированных точек в формате (x, y, z)
        '''
        x, y, z = center
        angles = np.linspace(0, 2 * np.pi, points_to_check, endpoint=False)
        points = [(x + step * np.cos(angle), y + step * np.sin(angle), z) for angle in angles]
        return points
    
    # Получаем максимальный размер модели и ее центр
    max_extent, center, _ = calculate_model_scale_and_camera_distance(mesh)
    
    # Инициализация переменных для поиска
    current_step = max_extent  # Начальный шаг поиска
    current_point = center  # Стартовая точка
    best_point = current_point  # Лучшая точка на текущий момент
    prev_dif = float('inf')  # Начальная разница для сравнения
    
    logging.info(f"Начальная точка: {current_point}, начальный шаг: {current_step}")
    
    # Основной цикл поиска
    while current_step > min_step:
        logging.info(f"Переход на новый шаг цикла с шагом: {current_step}")
        
        # Генерация точек на окружности вокруг текущей точки
        points = generate_points_in_circle(current_point, current_step, points_to_check)

        results = []  # Список для хранения результатов
        consecutive_increases = 0  # Счетчик последовательных увеличений разницы
        for point in points:
            # Оценка текущей точки
            logging.info(f"Оценка точки: x={point[0]:.3f}, y={point[1]:.3f}, z={point[2]:.3f}")
            
            # Перемещение модели в новую позицию
            image = move_model(*point, mesh)
            if image is None:
                continue
            
            # Преобразование изображения в grayscale и вычисление разницы
            model_image = np.array(image.convert('L'))
            dif, _ = difference(model_image, target_image)
            logging.info(f"Текущая разница для точки {point}: {dif}")
            
            if dif is not None:
                results.append((point, dif))
                # Проверка увеличения разницы
                if dif >= prev_dif:
                    consecutive_increases += 1
                    logging.info(f"Разница увеличилась: {consecutive_increases} последовательных увеличений")
                else:
                    consecutive_increases = 0  # Сбрасываем счетчик, если разница улучшилась
                prev_dif = dif
                
                # Прекращаем итерацию, если превышен порог увеличений
                if consecutive_increases >= threshold_points_to_stop:
                    logging.info(f"Превышен порог увеличений разницы. Завершаем текущую итерацию")
                    break

        # Если результатов нет, уменьшаем шаг
        if not results:
            current_step *= step_factor
            continue
        
        # Сортировка результатов по разнице
        results.sort(key=lambda x: x[1])
        
        # Выбор точки с минимальной разницей
        best_candidate, best_dif = results[0]
        
        if best_dif < min_dif:
            min_dif = best_dif
            best_point = best_candidate
            current_point = best_candidate
            logging.info(f"Обновлены лучшие точки: {best_point} с разницей: {min_dif}")
        else:
            # Уменьшаем шаг, так как дальнейшего улучшения не найдено
            logging.info(f"Дальнейшее улучшение не найдено, уменьшаем шаг")
            current_step *= step_factor
    
    logging.info("Завершение функции refine_search")
    return best_point, min_dif


def save_result_img(move_point, mesh):
    '''
    Функция для сохранения изображения 3D-модели, перемещенной в заданную точку

    Args:
        move_point (tuple): Координаты точки, в которую следует переместить модель (x, y, z)
        mesh (trimesh.Trimesh): 3D-модель, которая будет перемещена и рендерена

    Returns:
        None
    '''
    logging.info("Запуск функции save_result_img для сохранения изображения модели")
    
    # Извлекаем координаты точки
    x, y, z = move_point    
    # Перемещаем модель в заданную точку и сохраняем изображение
    image = move_model(x, y, z, mesh, save=True)
    
    # Указываем имя файла для сохранения
    file_name = './result.jpg'
    image.save(file_name, "JPEG")
    
    logging.info("Завершение функции save_result_img")
    return file_name

def load_mesh(input_obj_file):
    try:
        mesh = trimesh.load(input_obj_file)
        return mesh
    except Exception as e:
        raise ValueError(f"Ошибка при загрузке 3D модели: {e}")

def preprocess_mesh(mesh):
    '''
    Центрирует модель относительно начала координат
    '''
    mesh.apply_translation(-mesh.bounds.mean(axis=0))
    return mesh

def find_temp(input_img, input_obj_file):
    '''
    Основная функция для загрузки, обработки и рендеринга 3D-модели, а также поиска минимальной разницы с целевым изображением
    '''
    check_file_exists(input_obj_file)
    check_file_exists(input_img)

    mesh_or = load_mesh(input_obj_file)
    if mesh_or is None or not isinstance(mesh_or, trimesh.Trimesh):
        raise ValueError(f"Ошибка загрузки модели: {input_obj_file}")

    mesh_or = preprocess_mesh(mesh_or)
    target_image = align(remover(cv2.imread(input_img)))
    target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)
    
    min_dif = float('inf')
    angles, min_dif = render_and_save_image_parallel(mesh_or, target_image, min_dif)
    if angles is None or min_dif == float('inf'):
        raise RuntimeError("Не удалось найти оптимальные углы поворота модели")
    print(f"оптимальные углы поворота: {angles}, min_dif: {min_dif}")

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


    best_camera_angles, min_difference = refine_search(rotated_mesh, target_image, min_dif)
    if best_camera_angles is None or min_difference == float('inf'):
        raise RuntimeError("Ошибка в refine_search: не удалось найти лучшие параметры камеры")
    print(f"best_camera_angles: {best_camera_angles}, min_difference: {min_difference}, angles: {angles}, min_dif: {min_dif}")
    
    file_name = save_result_img(best_camera_angles, rotated_mesh)
    return file_name

# if __name__ == '__main__':
#     start_time = time.time()
#     main()
#     print(f"Время работы: {time.time() - start_time:.2f} секунд")
