import trimesh
import numpy as np
import time
import cv2
from compare import difference
from concurrent.futures import ProcessPoolExecutor
import pyrender
from PIL import Image, ImageOps


def rot(angle_x, angle_y, angle_z, mesh, invert_colors=True):
    '''
    поворот 3D-модели на заданные углы вокруг осей X, Y, Z и рендерит изображение модели
    используется на первом этапе - проверка 6 сторон модели
    :param angle_x: угол поворота по оси X в градусах
    :param angle_y: угол поворота по оси Y в градусах
    :param angle_z: угол поворота по оси Z в градусах
    :param mesh: 3D-модель 
    :param invert_colors: флаг для инверсии цветов изображения
    :return: изображение модели, повернутой на заданный угол
    '''
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
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=5.0)  # Настраиваем интенсивность света
    light_pose = np.eye(4)  # Положение источника света (по умолчанию в центре)
    scene.add(light, pose=light_pose)
    _, _, distance_mesh = calculate_model_scale_and_camera_distance(rotated_mesh)

    # Камера с фиксированным углом обзора
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 2.0)
    camera_pose = np.array([
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, distance_mesh],
                [0.0, 0.0, 0.0, 1.0],
            ])
    scene.add(camera, pose=camera_pose)

    # Рендеринг сцены
    r = pyrender.OffscreenRenderer(1024, 768)  # Создаём рендерер для захвата изображения
    color, _ = r.render(scene)  # Рендерим сцену в изображение

    # Преобразование в изображение
    image = Image.fromarray(color)

    # Инверсия цветов, если флаг установлен
    if invert_colors:
        image = ImageOps.invert(image)

    # Очищаем рендерер
    r.delete()

    return image

def process_direction(args):
    '''
    Обрабатывает заданный угол поворота модели, 
    рендерит её и вычисляет разницу между результатом и целевым изображением
    :param args: Кортеж параметров, включающий:
                    angle: Углы поворота (X, Y, Z)
                    file_name: Имя файла для сохранения рендера
                    mesh: 3D-модель
                    target_image: Целевое изображение (grayscale)
                    min_dif: Текущая минимальная разница

    :return: Углы поворота (X, Y, Z), Разница между рендером 
    и целевым изображением (или минимальная разница, если текущая больше)
    '''
    angle, file_name, mesh, target_image, min_dif = args
    angle_x, angle_y, angle_z = angle

    image = rot(angle_x, angle_y, angle_z, mesh)

    if image is None:
        return None, min_dif
    
    # Сохраняем изображение в file_name
    image.save(file_name, "JPEG")
    # Загружаем сохраненное изображение как grayscale
    model_image = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
    # Вычисляем разницу между текущим изображением и целевым 
    dif,_ = difference(model_image, target_image)

    return angle, dif if dif is not None and dif < min_dif else min_dif

def render_and_save_image_parallel(mesh, target_image, min_dif):
    '''
    Параллельно обрабатывает набор углов поворота модели и выбирает углы с минимальной разницей
    :param mesh: 3D-модель
    :param target_image: Целевое изображение
    :param min_dif: Текущая минимальная разница

    :return: Углы с минимальной разницей, Минимальная разница
    '''
    # Задаем углы для рендеринга
    directions = [
        ((0, 0, 0), './sides/front.jpg'),
        ((180, 0, 0), './sides/back.jpg'),
        ((90, 0, 0), './sides/top.jpg'),
        ((270, 0, 0), './sides/bottom.jpg'),
        ((0, 90, 0), './sides/right.jpg'),
        ((0, -90, 0), './sides/left.jpg')
    ]
    best_angles = (0, 0, 0)

    # запускаем функцию process_direction для каждого набора углов параллельно
    with ProcessPoolExecutor() as executor:
        results = executor.map(
            process_direction,
            [(angle, file_name, mesh, target_image, min_dif) for angle, file_name in directions]
        )
        # Обновляем минимальную разницу и лучшие углы поворота, если текущий результат лучше
        for angle, dif in results:
            print(angle, dif)
            if dif < min_dif:
                min_dif = dif
                best_angles = angle

    return best_angles, min_dif

def calculate_model_scale_and_camera_distance(mesh, yfov=np.pi / 3.0):
    '''
    Рассчитывает масштаб модели и минимальное расстояние камеры для полного обзора
    :param mesh: 3D-модель
    :param yfov: Поле зрения камеры (в радианах)
    :return: масштаб модели, минимальное расстояние камеры
    '''
    # Получаем размеры модели
    bounding_box = mesh.bounding_box.extents  # Длина, ширина, высота
    max_extent = max(bounding_box)  # Максимальный размер вдоль осей
    center = mesh.bounding_box.centroid  # Центр модели

    # Минимальное расстояние камеры для полного обзора модели
    distance = (3*max_extent+0.4)/2

    return max_extent, center, distance

def move_model(x, y, z, mesh, invert_colors=True, save=False):
    '''
    Перемещает модель на заданное смещение в пространстве и рендерит её изображение 
    (по сути, то же, что и rot, но, вместо поворота модели вокруг своей оси, 
    двигаем ее по координатной плоскости XY, чтобы найти нужный угол)

    :param x, y, z: координаты смещения модели
    :param mesh: 3D-модель
    :return: изображение модели с новой позиции
    '''
    # Создаем матрицу трансляции для смещения модели и применяем ее к копии модели
    translation_matrix = trimesh.transformations.translation_matrix([x, y, z])
    moved_mesh = mesh.copy()
    moved_mesh.apply_transform(translation_matrix)

    # Создаем сцену
    scene = pyrender.Scene()

    # Преобразование модели для использования с pyrender и добавление модели на сцену
    pyrender_mesh = pyrender.Mesh.from_trimesh(moved_mesh, smooth=True)
    scene.add(pyrender_mesh)
    
    # Настройка освещения
    if (x == 0 and y == 0):
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=5.0)
    else:
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=50.0) 
    light_pose = np.eye(4)  # Свет в фиксированной позиции (в центре сцены)
    scene.add(light, pose=light_pose)

    # Рассчитываем масштаб и расстояние камеры
    _, _, distance_mesh = calculate_model_scale_and_camera_distance(moved_mesh)
    if save: 
        distance_mesh = 2.5
        
    # Камера с фиксированным углом обзора
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 2.0)
    camera_pose = np.array([
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, distance_mesh],
                [0.0, 0.0, 0.0, 1.0],
            ])
    scene.add(camera, pose=camera_pose)

    # Рендеринг сцены
    r = pyrender.OffscreenRenderer(1024, 768)  # Создаем рендерер для захвата изображения
    color, _ = r.render(scene)  # Рендерим сцену в изображение

    # Преобразование результата в изображение
    image = Image.fromarray(color)
    if invert_colors:
        image = ImageOps.invert(image)
    # Очищаем рендерер
    r.delete()

    return image

def refine_search(mesh, target_image, min_dif, min_step=0.1, step_factor=0.5, points_to_check=5, threshold_points_to_stop=3):
    '''
    Реализация итеративного поиска минимальной точки с досрочным завершением итерации.
    
    :param mesh: 3D-модель
    :param target_image: Целевое изображение
    :param min_dif: Текущая минимальная разница
    :param min_step: Минимальный шаг для остановки поиска
    :param step_factor: Фактор уменьшения шага на каждой итерации
    :param points_to_check: Число точек для проверки в первой итерации
    :param threshold_points_to_stop: Число последовательных увеличений разницы для прекращения итерации
    :return: Координаты минимальной точки и минимальная разница
    '''
    def generate_points_in_circle(center, step, points_to_check):
        '''Генерация точек по кругу вокруг заданного центра.'''
        x, y, z = center
        angles = np.linspace(0, 2 * np.pi, points_to_check, endpoint=False)  # Углы круга
        points = []
        for angle in angles:
            dx = step * np.cos(angle)
            dy = step * np.sin(angle)
            points.append((x + dx, y + dy, z))  # Точки на окружности
        return points


    max_extent, center, _ = calculate_model_scale_and_camera_distance(mesh)
    current_step = max_extent # начальный шаг 
    current_point = center  # Стартовая точка
    best_point = current_point # лучшая точка на данный момент
    prev_dif = float('inf')  # Начальное значение для сравнения разницы
    print('center: ', center)
    while current_step > min_step:
        print(current_step)
        # Генерация точек на окружности вокруг текущей точки
        points = generate_points_in_circle(current_point, current_step, points_to_check)
        results = []
        consecutive_increases = 0  # Счетчик увеличений разницы
            
        for point in points:
            # Оцениваем текущую точку
            print(f"Текущая точка: x={point[0]:.3f}, y={point[1]:.3f}, z={point[2]:.3f}")
            # сдвигаем 3D-модель в новую позицию, соответствующую текущей точке
            image = move_model(*point, mesh)
            if image is None:
                continue
            # изображение преобразуется в градации серого
            model_image = np.array(image.convert('L'))
            file_name = 'notAresult.jpg'
            image.save(file_name, "JPEG")
            # Загружаем сохраненное изображение как grayscale
            model_image = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
            # разница между изображением модели и целевым изображением
            dif, _ = difference(model_image, target_image)
            print(dif)
            if dif is not None:
                results.append((point, dif))
                
                # Проверка последовательного увеличения разницы
                if dif >= prev_dif:
                    consecutive_increases += 1 # Если разница увеличилась, увеличивается счетчик
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

def save_result_img(move_point, mesh):
    x, y, z = move_point
    print(x,y,z)
    image = move_model(x, y, z, mesh, save=True)
    # Сохраняем изображение в file_name
    file_name = 'result.jpg'
    image.save(file_name, "JPEG")
    # Загружаем сохраненное изображение как grayscale
    model_image = cv2.imread(file_name, cv2.IMREAD_ANYCOLOR)
    cv2.imshow('model_image', model_image)
    cv2.waitKey(0)


if __name__ == '__main__':
    
    input_img = r'./od_their_ph.jpg'
    input_obj_file = r'./od_their.obj'

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
    print(f"result точка: x={best_camera_angles[0]:.3f}, y={best_camera_angles[1]:.3f}, z={best_camera_angles[2]:.3f}")
    save_result_img(best_camera_angles, rotated_mesh)
    
    # scene = trimesh.Scene(rotated_mesh)
    # scene.camera.resolution = (1024, 1024)
    # x, y, z = best_camera_angles
    # cam_rot = np.array([0, 0, 0])
    # center_point = np.array(rotated_mesh.centroid) + np.array([x, y, z]) 
    # print(center_point)
    # scene.set_camera(angles=cam_rot, distance=60, center=center_point)

    # output_obj_file_rotated = './rotated_model.obj'
    # rotated_mesh.export(output_obj_file_rotated)
    # finish = time.time()
    # print(f"Повернутая модель: {output_obj_file_rotated}")

    # rotated_mesh.show()

    # res_msec = (finish - start) * 1000
    # print('Время работы в миллисекундах: ', res_msec)

    # print("Изображения успешно сохранены.")
