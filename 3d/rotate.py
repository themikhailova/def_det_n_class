import trimesh
import numpy as np
import time
import cv2
from compare import difference
from concurrent.futures import ProcessPoolExecutor
import pyrender
from PIL import Image, ImageOps
import os
import logging

from det_aligning import align


logging.basicConfig(level=logging.INFO)

RENDER_WIDTH = 1024
RENDER_HEIGHT = 768
LIGHT_INTENSITY = 1.0
LIGHT_INTENSITY_HIGH = 5.0
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
    logging.info(f"Запуск функции rot для углов: X={angle_x}, Y={angle_y}, Z={angle_z}")

    # Преобразуем углы из градусов в радианы
    angle_x_rad = np.radians(angle_x)
    angle_y_rad = np.radians(angle_y)
    angle_z_rad = np.radians(angle_z)
    logging.debug(f"Углы в радианах: X={angle_x_rad}, Y={angle_y_rad}, Z={angle_z_rad}")

    # Создаем матрицы поворота для каждой оси
    rotation_matrix_x = trimesh.transformations.rotation_matrix(angle_x_rad, [1, 0, 0], mesh.centroid)
    rotation_matrix_y = trimesh.transformations.rotation_matrix(angle_y_rad, [0, 1, 0], mesh.centroid)
    rotation_matrix_z = trimesh.transformations.rotation_matrix(angle_z_rad, [0, 0, 1], mesh.centroid)
    logging.debug("Созданы матрицы поворота для осей X, Y, Z")

    # Комбинированная матрица поворота
    combined_rotation_matrix = np.dot(np.dot(rotation_matrix_x, rotation_matrix_y), rotation_matrix_z)
    logging.debug("Вычислена комбинированная матрица поворота")

    # Применяем поворот к копии модели
    rotated_mesh = mesh.copy()
    rotated_mesh.apply_transform(combined_rotation_matrix)
    logging.debug("Применен поворот к копии модели")

    # Создаем сцену для рендеринга
    scene = pyrender.Scene()
    trimesh_mesh = pyrender.Mesh.from_trimesh(rotated_mesh, smooth=True)
    scene.add(trimesh_mesh)
    logging.debug("Добавлена повернутая модель в сцену")

    # Настройка источника света
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=LIGHT_INTENSITY)
    light_pose = np.array([
        [1.0, 0.0, 0.0, 2.0],
        [0.0, 1.0, 0.0, 2.0],
        [0.0, 0.0, 1.0, 2.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    scene.add(light, pose=light_pose)
    logging.debug("Добавлен источник света в сцену")

    # Камера с фиксированным углом обзора
    camera = pyrender.PerspectiveCamera(yfov=CAMERA_FOV)
    camera_pose = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 2.5],
        [0.0, 0.0, 0.0, 1.0],
    ])
    scene.add(camera, pose=camera_pose)
    logging.debug("Добавлена камера в сцену")

    # Рендеринг сцены
    r = pyrender.OffscreenRenderer(RENDER_WIDTH, RENDER_HEIGHT)
    logging.info("Запущен рендеринг сцены")
    color, _ = r.render(scene)

    # Преобразование в изображение
    image = Image.fromarray(color)

    # Инверсия цветов, если флаг установлен
    if invert_colors:
        image = ImageOps.invert(image)
        logging.debug("Выполнена инверсия цветов изображения")

    # Очищаем рендерер
    r.delete()
    logging.debug("Очищен рендерер")

    # Проверка на пустое изображение
    if image is None or image.size == 0:
        logging.error("Ошибка рендеринга: получено пустое изображение")
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
    try:
        # Распаковка аргументов
        angle, file_name, mesh, target_image, min_dif = args
        angle_x, angle_y, angle_z = angle
        logging.info(f"Запуск process_direction для углов: X={angle_x}, Y={angle_y}, Z={angle_z}")

        # Рендеринг изображения с заданными углами
        image = rot(angle_x, angle_y, angle_z, mesh)
        if image is None:
            logging.warning(f"Не удалось сгенерировать изображение для углов: X={angle_x}, Y={angle_y}, Z={angle_z}")
            return None, min_dif

        # Сохранение рендеринга в файл
        image.save(file_name, "JPEG")
        logging.debug(f"Изображение сохранено в файл: {file_name}")

        # Загрузка сохраненного изображения в grayscale
        model_image = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
        if model_image is None:
            logging.error(f"Не удалось загрузить изображение из файла: {file_name}")
            return None, min_dif

        # Вычисление разницы между рендерингом и целевым изображением
        dif, _ = difference(model_image, target_image)
        if dif is None:
            logging.warning(f"Не удалось вычислить разницу для углов: X={angle_x}, Y={angle_y}, Z={angle_z}")
            return None, min_dif

        # Логирование результата вычисления разницы
        logging.info(f"Разница для углов X={angle_x}, Y={angle_y}, Z={angle_z}: {dif}")

    except Exception as e:
        logging.error(f"Ошибка в process_direction для углов X={angle_x}, Y={angle_y}, Z={angle_z}: {str(e)}")
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
    logging.debug(f"Определены направления для обработки: {directions}")

    best_angles = (0, 0, 0)  # Инициализация лучших углов поворота

    # Запускаем обработку углов параллельно
    try:
        with ProcessPoolExecutor() as executor:
            logging.info("Начата параллельная обработка углов")
            
            # Формируем аргументы для функции process_direction
            tasks = [(angle, file_name, mesh, target_image, min_dif) for angle, file_name in directions]
            results = executor.map(process_direction, tasks)
            
            # Обрабатываем результаты выполнения
            for angle, dif in results:
                if angle is None or dif is None:
                    logging.warning(f"Пропущен результат: угол={angle}, разница={dif}")
                    continue
                logging.debug(f"Обработан угол: {angle}, разница: {dif}")
                if dif < min_dif:
                    min_dif = dif
                    best_angles = angle
                    logging.info(f"Найдены новые лучшие углы: {best_angles} с разницей: {min_dif}")
    except Exception as e:
        logging.error(f"Ошибка во время параллельной обработки: {str(e)}")
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
        logging.debug("Извлечение размеров ориентированного bounding box модели")
        bounding_box = mesh.bounding_box_oriented.extents  # Длина, ширина, высота модели в её ориентации
        logging.debug(f"Размеры ориентированного bounding box: {bounding_box}")

        # Определение максимального размера среди осей (максимальная длина модели)
        max_extent = max(bounding_box)
        # Вычисление диагонали ограничивающего параллелепипеда
        diagonal = np.linalg.norm(bounding_box)  # Диагональ 3D-модели
        logging.debug(f"Диагональ модели: {diagonal}")

        # Вычисление центра модели (центроид модели)
        # Учёт смещения модели при вычислении её центра
        center = mesh.bounding_box.centroid
        # center[0] += offset[0]
        # center[1] += offset[1]
        # center[2] += offset[2]
        logging.debug(f"Центр модели с учётом смещения (центроид): {center}")

        aspect_ratio = render_width / render_height
        diagonal *= aspect_ratio 
        # Учитываем соотношение сторон кадра
        # 
        # if aspect_ratio > 1:
        #     diagonal *= aspect_ratio  # Горизонтальный кадр
        # else:
        #     diagonal /= aspect_ratio  # Вертикальный кадр

        # Расчёт минимального расстояния камеры с использованием диагонали
        logging.debug("Расчёт минимального расстояния камеры для полного охвата модели")
        distance = diagonal / (2 * np.tan(yfov / 2)) * safety_margin
        logging.debug(f"Рассчитанное минимальное расстояние камеры: {distance}")

        # Динамическая проверка попадания модели в кадр
        logging.debug("Проверка попадания модели в кадр")
        if not is_model_fully_in_frame(mesh, distance, yfov, render_width, render_height, offset):
            logging.debug("Модель не помещается в кадр, увеличение расстояния камеры")
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
    logging.debug("Создание трансляционной матрицы для перемещения модели")
    translation_matrix = trimesh.transformations.translation_matrix([x, y, z])
    moved_mesh = mesh.copy()
    moved_mesh.apply_transform(translation_matrix)
    logging.debug(f"Модель успешно смещена на ({x}, {y}, {z})")

    # Создаем сцену
    scene = pyrender.Scene()
    logging.debug("Создание сцены для рендеринга")

    # Преобразование модели для использования с pyrender и добавление модели на сцену
    pyrender_mesh = pyrender.Mesh.from_trimesh(moved_mesh, smooth=True)
    scene.add(pyrender_mesh)
    logging.debug("Модель добавлена в сцену")

    # Рассчитываем масштаб и расстояние камеры
    logging.debug("Рассчёт масштаба модели и минимального расстояния камеры")
    _, center, distance_mesh = calculate_model_scale_and_camera_distance(moved_mesh, offset=(x,y,z))
    center_x, center_y, _ = center
    # if save: 
    #     distance_mesh = 2.5 # для попадания в кадр крайних нужно минимум 2.5 но это слишком далеко для хорошего качества

    # Настройка освещения
    logging.debug("Настройка освещения")
    if (center_x == 0 and center_y == 0):
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=LIGHT_INTENSITY)
    else:
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=LIGHT_INTENSITY_HIGH) 
    light_pose = np.array([
        [1.0, 0.0, 0.0, 2.0],
        [0.0, 1.0, 0.0, 2.0],
        [0.0, 0.0, 1.0, 2.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    # scene.add(light, pose=light_pose)
    # light_pose = np.eye(4)  # Свет в фиксированной позиции (в центре сцены)
    scene.add(light, pose=light_pose)
    logging.debug("Освещение успешно добавлено")

    # Камера с фиксированным углом обзора
    logging.debug("Добавление камеры в сцену")
    camera = pyrender.PerspectiveCamera(yfov=CAMERA_FOV)
    camera_pose = np.array([
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, distance_mesh],
                [0.0, 0.0, 0.0, 1.0],
            ])
    scene.add(camera, pose=camera_pose)

    # Рендеринг сцены
    logging.debug("Рендеринг сцены для получения изображения")
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
            logging.debug("Применение инверсии цветов к изображению")
            image = ImageOps.invert(image)

        logging.debug("Применение автоконтраста к изображению для сохранения")
        image = ImageOps.autocontrast(image)

        logging.debug("Очистка рендерера после завершения рендеринга")
        r.delete()
        
    else:
        color, _ = r.render(scene)  # Рендерим сцену в изображение

        # Преобразование результата в изображение
        image = Image.fromarray(color)
        if invert_colors:
            logging.debug("Применение инверсии цветов к изображению")
            image = ImageOps.invert(image)
        image = ImageOps.autocontrast(image)
        # Очищаем рендерер
        logging.debug("Очистка рендерера после завершения рендеринга")
        r.delete()

    if image is None or image.size == 0:
        logging.error("Ошибка рендеринга: получено пустое изображение")
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
        logging.debug(f"Генерация точек по кругу вокруг центра {center} с шагом {step}")
        x, y, z = center
        angles = np.linspace(0, 2 * np.pi, points_to_check, endpoint=False)
        points = [(x + step * np.cos(angle), y + step * np.sin(angle), z) for angle in angles]
        logging.debug(f"Сгенерированные точки: {points}")
        return points
    
    # Получаем максимальный размер модели и ее центр
    logging.info("Получение максимального размера модели и центра")
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
                logging.warning(f"Не удалось получить изображение для точки {point}. Пропуск")
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
            logging.info("Результатов нет, уменьшаем шаг")
            current_step *= step_factor
            continue
        
        # Сортировка результатов по разнице
        results.sort(key=lambda x: x[1])
        
        # Выбор точки с минимальной разницей
        best_candidate, best_dif = results[0]
        logging.info(f"Лучшая точка после проверки: {best_candidate}, разница: {best_dif}")
        
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
    file_name = 'result.jpg'
    logging.debug(f"Сохраняем изображение в файл: {file_name}")
    image.save(file_name, "JPEG")

    # Загружаем изображение как цветное
    model_image = cv2.imread(file_name, cv2.IMREAD_ANYCOLOR)
    logging.debug(f"Изображение загружено, готово к отображению")
    
    # Отображаем изображение
    cv2.imshow('model_image', model_image)
    logging.info("Отображение изображения завершено. Ожидаем клавишу для закрытия")
    cv2.waitKey(0)
    
    logging.info("Завершение функции save_result_img")


def load_mesh(input_obj_file):
    '''
    Загружает 3D-модель из файла .obj
    
    Args:
        input_obj_file (str): Путь к файлу .obj с моделью
    
    Returns:
        trimesh.Trimesh: Загруженная 3D модель
    
    Raises:
        ValueError: Если не удалось загрузить модель
    '''
    try:
        mesh = trimesh.load(input_obj_file)
        return mesh
    except Exception as e:
        raise ValueError(f"Ошибка при загрузке 3D модели: {e}")

def preprocess_mesh(mesh):
    '''
    Центрирует модель относительно начала координат
    
    Args:
        mesh (trimesh.Trimesh): 3D-модель для обработки
    
    Returns:
        trimesh.Trimesh: Центрированная модель
    '''
    mesh.apply_translation(-mesh.bounds.mean(axis=0))
    return mesh

def main():
    '''
    Основная функция для загрузки, обработки и рендеринга 3D-модели, а также поиска минимальной разницы с целевым изображением
    '''
    input_img = r'./fig4.jpg'
    input_obj_file = r'./mod4.obj'
    
    check_file_exists(input_obj_file)
    check_file_exists(input_img)

    mesh_or = load_mesh(input_obj_file)
    mesh_or = preprocess_mesh(mesh_or)
    
    target_image = align(input_img)
    target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)

    # cv2.imshow('target', target_image)
    # cv2.waitKey(0)

    min_dif = float('inf')
    angles, min_dif = render_and_save_image_parallel(mesh_or, target_image, min_dif)
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
    print(f"best_camera_angles: {best_camera_angles}, min_difference: {min_difference}, angles: {angles}, min_dif: {min_dif}")
    
    save_result_img(best_camera_angles, rotated_mesh)

if __name__ == '__main__':
    start_time = time.time()
    main()
    print(f"Время работы: {time.time() - start_time:.2f} секунд")
