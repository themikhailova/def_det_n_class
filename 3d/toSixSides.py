import trimesh
import numpy as np
from PIL import Image
import io
import os
import time

input_obj_file = 'model1.stl'

def render_and_save_image(mesh):
    directions = [
        ((0, 0, 0), './model_sides1/front.jpg'),
        ((180, 0, 0), './model_sides1/back.jpg'),
        ((90, 0, 0), './model_sides1/top.jpg'),
        ((270, 0, 0), './model_sides1/bottom.jpg'),
        ((0, 90, 0), './model_sides1/right.jpg'),
        ((0, -90, 0), './model_sides1/left.jpg')
    ]

    for angle, file_name in directions:
        angle_x, angle_y, angle_z = angle
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

        # если модель слишком маленькая
        min_bound, max_bound = rotated_mesh.bounds
        if np.allclose(min_bound, max_bound):  # нулевые размеры
            print(f"Ошибка: размеры модели слишком малы или нулевые для поворота ({angle_x}, {angle_y}, {angle_z})")
            continue

        center_point = (min_bound + max_bound) / 2
        scene.camera.look_at([center_point], distance=2)
        # Рендеринг сцены
        image_data = scene.save_image(visible=True, background=[0, 0, 0, 0])
        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        image.save(file_name, "JPEG")


mesh = trimesh.load(input_obj_file)

start = time.time()
render_and_save_image(mesh)
finish = time.time()
res_msec = (finish - start) * 1000
print('Время работы в миллисекундах: ', res_msec)

print("Изображения успешно сохранены.")