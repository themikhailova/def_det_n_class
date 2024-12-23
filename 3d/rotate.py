import trimesh
import numpy as np
from PIL import Image
import io
import copy
import time
import cv2
from compare import difference
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import trimesh
import trimesh
import numpy as np
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
    light = pyrender.PointLight(color=np.ones(3), intensity=5.0)  # Настраиваем интенсивность света
    light_pose = np.eye(4)  # Положение источника света (по умолчанию в центре)
    scene.add(light, pose=light_pose)

    # Настройка камеры
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
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
        ((0, 0, 0), './front.jpg'),
        ((180, 0, 0), './back.jpg'),
        ((90, 0, 0), './top.jpg'),
        ((270, 0, 0), './bottom.jpg'),
        ((0, 90, 0), './right.jpg'),
        ((0, -90, 0), './left.jpg')
    ]
    best_angles = (0, 0, 0)

    with ProcessPoolExecutor() as executor:
        results = executor.map(
            process_direction,
            [(angle, file_name, mesh, target_image, min_dif) for angle, file_name in directions]
        )
        for angle, dif in results:
            if dif < min_dif:
                min_dif = dif
                best_angles = angle

    return best_angles, min_dif

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
        light = pyrender.PointLight(color=np.ones(3), intensity=500.0)  # Интенсивность света
    light_pose = np.eye(4)  # Свет в фиксированной позиции (в центре сцены)
    scene.add(light, pose=light_pose)

    # --- Настройка камеры ---
    bounding_box = moved_mesh.bounding_box.extents
    max_extent = max(bounding_box)  # Максимальное измерение модели
    distance_mesh = max_extent * 5  # Увеличиваем расстояние для большего обзора

    # Камера с фиксированным углом обзора
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)

    if x == 0 and y == 0:
        camera_pose = np.array([
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 2.5],
                [0.0, 0.0, 0.0, 1.0],
            ])
    else:
        camera_pose = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, distance_mesh],
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

def process_camera_rotation(args):
    angle, file_name, mesh, target_image, min_dif = args
    angle_x, angle_y, angle_z = angle
    image = move_model(angle_x, angle_y, angle_z, mesh)
    if image is None:
        return None, min_dif
    image.save(file_name, "JPEG")
    model_image = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
    dif,_ = difference(model_image, target_image)
    return angle, dif if dif is not None and dif < min_dif else None

def render_images_camera_rotation(mesh, target_image, min_dif):
    best_angles = (0, 0, 0)
    chunks = cut_directions()
    for direction in chunks:
        with ProcessPoolExecutor() as executor:
            results = executor.map(
                process_camera_rotation,
                [(angle, file_name, mesh, target_image, min_dif) for angle, file_name in direction]
            )
            for angle, dif in results:
                if dif is not None:
                    if dif < min_dif:
                        min_dif = dif
                        best_angles = angle

    return best_angles, min_dif

def full_directions():
    direction = []
    for x in range(-1, 2, 1):
        for y in range(-1, 2, 1):
            point = (x, y, 0)
            file_name = f'./{x}_{y}_0.jpg'
            dir = (point, file_name)
            direction.append(dir)
    return direction

def cut_directions():
    directions = full_directions()
    chunk_size = 4
    chunks = [directions[i:i + chunk_size] for i in range(0, len(directions), chunk_size)]
    return chunks


if __name__ == '__main__':
    
    input_img = r'./try11noBack.jpg'
    input_obj_file = r'./od1.obj'

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
    best_camera_angles, min_difference = render_images_camera_rotation(rotated_mesh, target_image, min_dif)
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
