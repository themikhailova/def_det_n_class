
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
import time 

def compare_neighboring_regions(image, region_size=0.05, threshold=10, contour_area_threshold=800): 
    """
    Сравнение соседних областей изображения для обнаружения аномалий с выделением контуров.
    """
    h, w = image.shape
    region_h, region_w = int(h * region_size), int(w * region_size)
    
    # Вычисление средней яркости и стандартного отклонения в каждой области
    anomaly_map = np.zeros_like(image, dtype=np.uint8)
    real_anomaly_map = np.zeros_like(image, dtype=np.uint8)
    # Обработка изображения для поиска аномальных областей
    for y in range(0, h, region_h):
        for x in range(0, w, region_w):
            # Текущая область
            region = image[y:y+region_h, x:x+region_w]
            mean_region = np.mean(region)
            
            # Сравнение с соседними областями
            neighbors = []
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    ny, nx = y + dy * region_h, x + dx * region_w
                    if 0 <= ny < h and 0 <= nx < w:
                        neighbor_region = image[ny:ny+region_h, nx:nx+region_w]
                        neighbors.append(np.mean(neighbor_region))
            
            if neighbors:
                mean_neighbors = np.mean(neighbors)
                if abs(mean_region - mean_neighbors) > threshold:
                    # Обозначение области как аномальной
                    anomaly_map[y:y+region_h, x:x+region_w] = 255

    # Найдем контуры на исходном изображении в аномальных областях, помеченных в anomaly_map
    result_image = np.copy(image)  # Копия исходного изображения для рисования

    # Применяем маску anomaly_map для выделения только аномальных областей
    masked_image = cv2.bitwise_and(result_image, result_image, mask=anomaly_map)
    
    # Применяем алгоритм Кэнни для поиска краев в аномальных областях
    edges = cv2.Canny(masked_image, 100, 200)  # Алгоритм Кэнни для поиска краев

    # Поиск контуров на изображении с краями
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Рисуем контуры, пропуская слишком большие (по площади) прямоугольные контуры
    for contour in contours:
        # Вычисляем площадь контура
        area = cv2.contourArea(contour)
        if area < contour_area_threshold:  # Фильтруем контуры по площади
            cv2.drawContours(real_anomaly_map, [contour], -1, (0, 0, 255), 2)  # Красный цвет для контуров
    
    return result_image, real_anomaly_map
    

def detect_anomalies_with_dbscan(diff_image, eps=3, min_samples=10):
    """
    Обнаружение аномалий с помощью DBSCAN.
    """
    # Преобразование карты разницы в массив координат
    coords = np.column_stack(np.where(diff_image > 0))
    
    if len(coords) == 0:
        return diff_image  # Если нет аномалий, возвращаем пустую карту
    
    # Кластеризация аномальных пикселей
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
    labels = clustering.labels_

    # Создание карты кластеров
    clustered_anomalies = np.zeros_like(diff_image, dtype=np.uint8)
    for label in set(labels):
        if label == -1:
            continue  # Пропускать шум (точки вне кластеров)
        cluster_coords = coords[labels == label]
        for y, x in cluster_coords:
            clustered_anomalies[y, x] = 255

    return clustered_anomalies

def merge_close_contours(anomaly_map, dilate_iter=1):
    """
    Объединение близко расположенных контуров в один общий контур.
    """
    # Расширение контуров
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(anomaly_map, kernel, iterations=dilate_iter)

    # Поиск объединенных контуров
    merged_contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return merged_contours

def detect_anomalies(input_image, reference_image, threshold=60, region_size=0.05):
    """
    Основной алгоритм обнаружения аномалий.
    """
    # Преобразование в градации серого
    input_gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    reference_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
    
    # Вычисление разницы между изображениями
    diff = cv2.absdiff(input_gray, reference_gray)
    
    # Применение порога для выделения аномалий
    _, anomalies = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    # Обнаружение аномалий на основе соседних областей
    _, region_anomalies = compare_neighboring_regions(input_gray, region_size=region_size)

    # Комбинирование результатов
    combined_anomalies = cv2.bitwise_or(anomalies, region_anomalies)

    # Использование DBSCAN для выделения кластеров аномалий
    # clustered_anomalies = detect_anomalies_with_dbscan(combined_anomalies)
    
    # Объединение близко расположенных контуров
    # merged_contours = merge_close_contours(combined_anomalies)
    
    # # Рисование итоговых контуров на изображении
    # output_image = input_image.copy()
    # cv2.drawContours(output_image, merged_contours, -1, (0, 0, 255), 2)  # Красный цвет для контуров

    # Поиск контуров для визуализации
    contours, _ = cv2.findContours(combined_anomalies, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Рисование контуров на изображении
    output_image = input_image.copy()
    cv2.drawContours(output_image, contours, -1, (0, 0, 255), 2)  # Красный цвет для контуров

    return output_image, combined_anomalies

# Загрузка изображений
input_image = cv2.imread("./ourdets/blue/refAnddef/def.jpg")
reference_image = cv2.imread("./ourdets/blue/refAnddef/1/ref.jpg")
start = time.time()
# Обнаружение аномалий
output_image, anomalies = detect_anomalies(input_image, reference_image)
finish = time.time()
res_msec = (finish - start) * 1000
print('Время работы в миллисекундах: ', res_msec)
# Сохранение результатов
cv2.imwrite("output_image.png", output_image)
cv2.imwrite("anomalies.png", anomalies)
