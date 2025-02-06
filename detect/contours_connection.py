import cv2
import numpy as np
from scipy.spatial.distance import cdist
import sys, os
sys.path.append(os.path.abspath('./detect'))
from features import calculate_features

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

        features = calculate_features(contour, input_gray, max_area, max_perimeter, max_centroid)
        if features['mean_intensity'] < 40:
            continue
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
                diff_centroid = abs(float(contour_info['features']['relative_centroid_distance']) - float(other_contour_info['features']['relative_centroid_distance']))
                diff_min_intensivity = abs(float(contour_info['features']['min_intensity']) - float(other_contour_info['features']['min_intensity']))
                diff_mean_gradient = abs(float(contour_info['features']['mean_gradient']) - float(other_contour_info['features']['mean_gradient']))
                diff_std_gradient = abs(float(contour_info['features']['std_gradient']) - float(other_contour_info['features']['std_gradient']))
                # если разница не превосходит установленный порог 
                if (diff_intensity < intensity_threshold and 
                        diff_centroid < centroid_threshol and 
                        diff_min_intensivity < min_intensivity_threshol and 
                        diff_mean_gradient < mean_gradien_threshol and 
                        diff_std_gradient < std_gradient_threshol):
                        
                        # если это контур еще не был ни с кем объединен
                        if other_contour_info['used'] == False:         
                            # объединяем сначала оба массива точек в один, потом обводим минимальным выпуклым контуром               
                            merged_contour = merge_contours(merged_contour, other_contour_info['contour'])
                            
                            cv2.drawContours(input_gray_copy, [merged_contour], -1, (0, 0, 255), 2)
                            other_contour_info['used'] = True
                            used_indices.add(other_contour_info['num'])
        merged_contours.append(merged_contour)
        used_indices.add(idx)
        
    # Возвращаем объединенные контуры
    return merged_contours if merged_contours else contours

def merge_overlapping_contours(contours, input_gray, max_area, max_perimeter, max_centroid, intensity_threshold=55, centroid_threshol=35, min_intensivity_threshol=35, max_intensity_threshol=35, mean_gradien_threshol=15, std_gradient_threshol=35):
    import networkx as nx

    def rectangles_intersect(rect1, rect2):
        x1, y1, w1, h1 = rect1
        x2, y2, w2, h2 = rect2
        return not (x1 + w1 <= x2 or x2 + w2 <= x1 or y1 + h1 <= y2 or y2 + h2 <= y1)

    # Вычисляем boundingRect и признаки для всех контуров
    bounding_boxes = [cv2.boundingRect(contour) for contour in contours]
    features = [calculate_features(contour, input_gray, max_area, max_perimeter, max_centroid) for contour in contours]

    # Строим граф
    G = nx.Graph()
    G.add_nodes_from(range(len(contours)))

    for i, rect1 in enumerate(bounding_boxes):
        for j, rect2 in enumerate(bounding_boxes[i + 1:], start=i + 1):
            if rectangles_intersect(rect1, rect2):
                diff_intensity = abs(features[i]['mean_intensity'] - features[j]['mean_intensity'])
                diff_centroid = abs(float(features[i]['relative_centroid_distance']) - float(features[j]['relative_centroid_distance']))
                diff_min_intensivity = abs(float(features[i]['min_intensity']) - float(features[j]['min_intensity']))
                diff_max_intensity = abs(float(features[i]['max_intensity']) - float(features[j]['max_intensity']))
                diff_mean_gradient = abs(float(features[i]['mean_gradient']) - float(features[j]['mean_gradient']))
                diff_std_gradient = abs(float(features[i]['std_gradient']) - float(features[j]['std_gradient']))
                    # если разница не превосходит установленный порог 
                if (diff_intensity < intensity_threshold and 
                        diff_centroid < centroid_threshol and 
                        diff_min_intensivity < min_intensivity_threshol and 
                        diff_max_intensity < max_intensity_threshol and 
                        diff_mean_gradient < mean_gradien_threshol and 
                        diff_std_gradient < std_gradient_threshol):
                    G.add_edge(i, j)

    # Находим компоненты связности и объединяем контуры
    merged_contours = []
    for component in nx.connected_components(G):
        merged_contour = np.vstack([contours[i] for i in component])
        merged_contours.append(merged_contour)

    return merged_contours



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
                if (diff_intensity < intensity_threshold and 
                    diff_min_intensivity < min_intensivity_threshol and 
                    diff_max_intensity < max_intensity_threshol):
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
