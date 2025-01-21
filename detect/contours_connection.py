import cv2
import numpy as np
from scipy.spatial.distance import cdist

from features import calculate_features

from shapely.geometry import Polygon, GeometryCollection

def find_nearest_points(contour1, contour2):
    # Преобразуем контуры в двумерные массивы точек
    points1 = contour1.reshape(-1, 2)
    points2 = contour2.reshape(-1, 2)

    # Вычисляем попарные расстояния между точками двух контуров
    distances = cdist(points1, points2)

    # Находим индексы минимального расстояния
    min_index = np.unravel_index(np.argmin(distances), distances.shape)
    point1 = points1[min_index[0]]
    point2 = points2[min_index[1]]

    return point1, point2

def merge_contours(contour1, contour2):
    # Найти ближайшие точки между двумя контурами
    point1, point2 = find_nearest_points(contour1, contour2)

    # Создать новый контур, соединяя исходные контуры через ближайшие точки
    index1 = np.where((contour1[:, 0] == point1).all(axis=1))[0][0]
    index2 = np.where((contour2[:, 0] == point2).all(axis=1))[0][0]

    # Объединяем контуры
    merged_contour = np.vstack((
        contour1[:index1+1],  # Часть первого контура до ближайшей точки
        contour2[index2:],   # Часть второго контура после ближайшей точки
        contour2[:index2+1], # Часть второго контура до ближайшей точки
        contour1[index1:]    # Часть первого контура после ближайшей точки
    ))

    return merged_contour

def calculate_min_distance(contour1, contour2):
        """Вычисляет минимальное расстояние между двумя контурами."""
        # Преобразуем контуры в массивы координат
        points1 = np.array([pt[0] for pt in contour1], dtype=float)
        points2 = np.array([pt[0] for pt in contour2], dtype=float)
        
        # Вычисляем матрицу расстояний
        distances = cdist(points1, points2, metric='euclidean')
        
        # Возвращаем минимальное расстояние
        return np.min(distances)

def connect_contours(contours, input_gray, max_area, max_perimeter, max_centroid, 
                     intensity_threshold=25, concavity_threshold=10, complexity_threshold=0.3, entropy_threshold=5, std_intensity_threshold=5,num_nearest=5, min_dist=15,
                     centroid_threshol=45, min_intensivity_threshol=35, max_intensity_threshol=55, mean_gradien_threshol=35, std_gradient_threshol=25):
    i = 0
    dict_cont = []
    input_gray_copy = input_gray.copy()
    # собираем список словарей, включающий в себя контур, его характеристики и флаг использования
    for contour in contours:
        if cv2.contourArea(contour) < 15:
            continue
        # if len(contour) < 5:
        #     continue
        features = calculate_features(contour, input_gray, max_area, max_perimeter, max_centroid)
        # if features['relative_area'] > 0.05:
        #     continue
        dict_cont.append({'num': i, 'features': features, 'contour': contour, 'used': False})  
        i += 1

    # список для объединенных контуров и множества использованных индексов
    merged_contours = []
    used_indices = set()

    # рассматриваем каждый контур на наличие для него ближайших 
    for idx, contour_info in enumerate(dict_cont):
    
        # Пропускаем контур, если он уже использован
        if contour_info['used']:
            continue

        closest_contours = []  # список ближайших контуров
        for jdx, other_contour_info in enumerate(dict_cont):
            # Пропускаем, если другой контур уже использован
            if other_contour_info['used']:
                continue
                
            # расстояние между основным контуром и другим
            distance = calculate_min_distance(contour_info['contour'], other_contour_info['contour'])
            closest_contours.append({'idx': jdx, 'contour': other_contour_info, 'distance': distance}) # каждый контур добавляем в список с информацией о расстоянии до него от основного

        # сортируем ближайшие контуры по расстоянию
        closest_contours = sorted(closest_contours, key=lambda x: x['distance'])
        merged_contour = contour_info['contour']
        
        # рассматриваем каждый ближайший контур
        for nearest in closest_contours:
            other_contour_info = nearest['contour']
            distance = nearest['distance']
            
            # проверяем минимальное расстояние
            if distance < min_dist:
                diff_intensity = abs(contour_info['features']['mean_intensity'] - other_contour_info['features']['mean_intensity'])
                # diff_entropy = abs(contour_info['features']['entropy'] - other_contour_info['features']['entropy'])
                # diff_std_intensity = abs(contour_info['features']['std_intensity'] - other_contour_info['features']['std_intensity'])

                diff_centroid = abs(float(contour_info['features']['relative_centroid_distance']) - float(other_contour_info['features']['relative_centroid_distance']))
                diff_min_intensivity = abs(float(contour_info['features']['min_intensity']) - float(other_contour_info['features']['min_intensity']))
                diff_max_intensity = abs(float(contour_info['features']['max_intensity']) - float(other_contour_info['features']['max_intensity']))
                diff_mean_gradient = abs(float(contour_info['features']['mean_gradient']) - float(other_contour_info['features']['mean_gradient']))
                diff_std_gradient = abs(float(contour_info['features']['std_gradient']) - float(other_contour_info['features']['std_gradient']))
                # если разница не превосходит установленный порог 
                if (diff_intensity < intensity_threshold and 
                    # diff_entropy < entropy_threshold and 
                    # diff_std_intensity < std_intensity_threshold and 
                        diff_centroid < centroid_threshol and 
                        diff_min_intensivity < min_intensivity_threshol and 
                        # diff_max_intensity < max_intensity_threshol and 
                        diff_mean_gradient < mean_gradien_threshol and 
                        diff_std_gradient < std_gradient_threshol):
                        # print('contour_info: ', contour_info['features'])
                        # print('other_contour_info: ',  other_contour_info['features'])
                    # если это контур еще не был ни с кем объединен
                        if other_contour_info['used'] == False:         
                            # объединяем сначала оба массива точек в один, потом обводим минимальным выпуклым контуром               
                            merged_contour = merge_contours(merged_contour, other_contour_info['contour'])
                            
                            cv2.drawContours(input_gray_copy, [merged_contour], -1, (0, 0, 255), 2)
                            other_contour_info['used'] = True
                            used_indices.add(other_contour_info['num'])
        merged_contours.append(merged_contour)
        used_indices.add(idx)
        
                        
                # else:
                #     merged_contours.append(contour_info['contour'])
                
        # Если контур не был объединен, добавляем его в итоговый список
        # if idx not in used_indices:
        #     merged_contours.append(contour_info['contour'])
    print('2')
    # cv2.imshow('adsf', input_gray_copy)
    # cv2.waitKey(0)
    cv2.imwrite('./connectn.jpg', input_gray_copy)
    print('3')
    # Возвращаем объединенные контуры
    return merged_contours if merged_contours else contours

def are_contours_intersecting(contour1, contour2):
    """
    Проверяет, пересекаются ли два контура contour1 и contour2.
    
    :param contour1: Контур первого объекта.
    :param contour2: Контур второго объекта.
    :return: True, если контуры пересекаются; False в противном случае.
    """
    # Проверка на пересечения контуров
    intersection = cv2.intersectConvexConvex(contour1, contour2)
    print(intersection[0] )
    if intersection[0] > 0:  # Если площадь пересечения больше нуля, значит пересекаются
        return True
    return False

# def get_ellipse_params(contour):
#     """ 
#     Получает параметры эллипса, описывающего контур. 
#     """
#     (x, y), (MA, ma), angle = cv2.fitEllipse(contour)
#     return (x, y), MA / 2, ma / 2, angle 

# def are_ellipses_intersecting(ellipse1, ellipse2):
#     """ 
#     Проверяет, пересекаются ли два эллипса. 
#     """
#     center1, axes11, axes12, angle1 = ellipse1
#     center2, axes21, axes22, angle2 = ellipse2
    
#     # Генерируем точки по эллипсам
#     points1 = cv2.ellipse2Poly((int(center1[0]), int(center1[1])), (int(axes11), int(axes12)), int(angle1), 0, 360, 1)
#     points2 = cv2.ellipse2Poly((int(center2[0]), int(center2[1])), (int(axes21), int(axes22)), int(angle2), 0, 360, 1)
    
#     # Проверяем пересечение
#     return cv2.intersectConvexConvex(points1, points2)[0] > 0

# def are_contours_intersecting(contour1, contour2):
#     """ 
#     Проверяет, пересекаются ли два контура. 
#     """
#     ellipse1 = get_ellipse_params(contour1)
#     ellipse2 = get_ellipse_params(contour2)
    
#     return are_ellipses_intersecting(ellipse1, ellipse2)

def union_contours(contours, input_gray, max_area, max_perimeter, max_centroid):
    """
    
    :param contours: Список контуров
    :return: Список контуров
    """
    merged_contours = []
    used_indices = set()  # Множество для хранения индексов удаленных контуров
    input_gray_copy = input_gray.copy()
    for i, outer_contour in enumerate(contours):
        if i in used_indices:  # Пропускаем контуры, которые уже были объединены
            continue
        
        merged_contour = outer_contour  # Начинаем с текущего внешнего контура
        for j, inner_contour in enumerate(contours):
            if i != j and are_contours_intersecting(outer_contour, inner_contour):
                features1 = calculate_features(outer_contour, input_gray, max_area, max_perimeter, max_centroid)
                features2 = calculate_features(inner_contour, input_gray, max_area, max_perimeter, max_centroid)
                diff_intensity = abs(features1['mean_intensity'] - features2['mean_intensity'])
                diff_concavity = abs(features1['concavity'] - features2['concavity'])
                diff_complexity = abs(features1['complexity'] - features2['complexity'])

                if (diff_intensity < 30 and 
                    diff_concavity < 20 and 
                    diff_complexity < 20):
                    # merged_contour = cv2.convexHull(np.vstack((merged_contour, inner_contour)))
                    merged_contour = merge_contours(outer_contour, inner_contour)
                    used_indices.add(j)  # Помечаем внутренний контур как использованный
        merged_contours.append(merged_contour)
                
           
            
        # else:
        #     merged_contours.append(merged_contour)
             # Добавляем объединенный контур, если он был изменен
    cv2.drawContours(input_gray_copy, [merged_contour], -1, (0, 0, 0), 2)
        # used_indices.add(i)
    print('merged_contour: ',len(merged_contours))
    cv2.imshow('adsf', input_gray_copy)
    cv2.waitKey(0)   
    
    
    # Добавляем только те контуры, которые не были объединены
    # remaining_contours = [contour for i, contour in enumerate(contours) if i not in used_indices]
    # print(len(contours))
    # print(len(merged_contours) + len(remaining_contours))
    print('used_indices: ', len(used_indices))
    if len(contours)!=len(merged_contours) :
        flag = True
    else:
        flag = False
    return merged_contours, flag

def is_contour_inside(outer_contour, inner_contour):
    """
    Проверяет, является ли контур inner_contour вложенным в контур outer_contour.
    
    :param outer_contour: Контур внешнего объекта (больший контур)
    :param inner_contour: Контур внутреннего объекта (меньший контур)
    :return: True, если контур inner_contour полностью вложен в outer_contour; False в противном случае.
    """
    for point in inner_contour:
        x, y = int(point[0][0]), int(point[0][1])  # Точка в контуре — это массив вида [x, y]            
        if cv2.pointPolygonTest(outer_contour, (x, y), False) < 0:
            return False  # Если хотя бы одна точка не внутри, возвращаем False
    return True  # Если все точки внутри, возвращаем True

def increase_ellipse(ellipse, scale_factor=1.3):
    center, axes, angle = ellipse
    major_axis, minor_axis = axes

    new_major_axis = major_axis * scale_factor
    new_minor_axis = minor_axis * scale_factor

    return (center, (new_major_axis, new_minor_axis), angle)

def is_contour_inside_ellipse(outer_contour, inner_contour):
    # Получаем параметры эллипса
    (x, y), (major_axis, minor_axis), angle = cv2.fitEllipse(outer_contour)

    # Проверяем, что все точки внутреннего контура находятся внутри эллипса
    for point in inner_contour:
        point = point[0]  # Преобразуем в нужный формат
        # Преобразуем координаты точки относительно центра эллипса
        normalized_x = (point[0] - x) * np.cos(np.radians(angle)) + (point[1] - y) * np.sin(np.radians(angle))
        normalized_y = -(point[0] - x) * np.sin(np.radians(angle)) + (point[1] - y) * np.cos(np.radians(angle))

        # Проверяем, попадает ли точка в уравнение эллипса
        if (normalized_x ** 2) / (major_axis / 2) ** 2 + (normalized_y ** 2) / (minor_axis / 2) ** 2 > 1:
            return False  # Точка вне эллипса

    return True  # Все точки внутри эллипса
def is_contour_within_bounding_rect(outer_contour, inner_contour):
    """
    Проверяет, находится ли меньший контур в рамках ограничивающего прямоугольника большого контура.
    
    :param outer_contour: Контур, который рассматривается как внешний
    :param inner_contour: Контур, который рассматривается как вложенный
    :return: True, если внутренний контур находится в пределах ограничивающего прямоугольника большого контура
    """
    # Находим ограничивающий прямоугольник для внешнего контура
    x_outer, y_outer, w_outer, h_outer = cv2.boundingRect(outer_contour)
    
    # Находим ограничивающий прямоугольник для внутреннего контура
    x_inner, y_inner, w_inner, h_inner = cv2.boundingRect(inner_contour)

    # Проверяем, находится ли внутренний прямоугольник в пределах внешнего
    return (x_inner >= x_outer and 
            y_inner >= y_outer and 
            x_inner + w_inner <= x_outer + w_outer and 
            y_inner + h_inner <= y_outer + h_outer)    

def remove_nested_contours(contours, input_gray, max_area, max_perimeter, max_centroid, intensity_threshold=50, min_intensivity_threshol=20, max_intensity_threshol=20):
    contours_to_remove = set()  # Множество контуров для удаления

    for i, outer_contour in enumerate(contours):
        if i in contours_to_remove:  # Пропускаем контуры, которые уже помечены для удаления
            continue
        
        for j, inner_contour in enumerate(contours):
            if i != j and is_contour_within_bounding_rect(outer_contour, inner_contour):
                features1 = calculate_features(outer_contour, input_gray, max_area, max_perimeter, max_centroid)
                features2 = calculate_features(inner_contour, input_gray, max_area, max_perimeter, max_centroid)
                # diff_intensity = abs(features1['mean_intensity'] - features2['mean_intensity'])
                # diff_concavity = abs(features1['concavity'] - features2['concavity'])
                # diff_complexity = abs(features1['complexity'] - features2['complexity'])
                diff_area = abs(features1['area'] - features2['area'])

                # # Параметры объединения контуров
                # if (diff_intensity < 50 and 
                #     diff_concavity < 30 and 
                #     diff_complexity < 20):
                diff_intensity = abs(features1['mean_intensity'] - features2['mean_intensity'])
                diff_entropy = abs(features1['entropy'] - features2['entropy'])
                diff_std_intensity = abs(features1['std_intensity'] - features2['std_intensity'])
                diff_min_intensivity = abs(float(features1['min_intensity']) - float(features2['min_intensity']))
                diff_max_intensity = abs(float(features1['max_intensity']) - float(features2['max_intensity']))
                # если разница между средней яркостью, вокнутостью и сложностью контуров не превосходит установленный порог 
                # if (diff_intensity < intensity_threshold and 
                #     diff_min_intensivity < min_intensivity_threshol and 
                #     diff_max_intensity < max_intensity_threshol):
                outer_contour = merge_contours(outer_contour, inner_contour)
                contours_to_remove.add(j)  # Добавляем индекс вложенного контура
                # elif (diff_area > 40):
                #     outer_contour = merge_contours(outer_contour, inner_contour)
                #     contours_to_remove.add(j)

        # Обновляем контур, после окончания цикла
        contours[i] = outer_contour

    # Оставляем только те контуры, которые не были помечены для удаления
    remaining_contours = [contour for i, contour in enumerate(contours) if i not in contours_to_remove]

    return remaining_contours
