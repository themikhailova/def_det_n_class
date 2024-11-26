import cv2
import numpy as np
import time

# Функция для расчета характеристик
def calculate_features(contour, image_gray):
    features = {}

    # Площадь и периметр
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    features['area'] = area
    features['perimeter'] = perimeter

    # Компактность
    if area > 0:
        compactness = (perimeter ** 2) / (4 * np.pi * area)
        features['compactness'] = compactness
    else:
        features['compactness'] = 0

    # Ограничивающий прямоугольник
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = w / h
    features['aspect_ratio'] = aspect_ratio

    # Эллипс (если достаточно точек)
    if len(contour) >= 5:
        ellipse = cv2.fitEllipse(contour)
        (center, axes, angle) = ellipse
        major_axis = max(axes)
        minor_axis = min(axes)
        eccentricity = np.sqrt(1 - (minor_axis**2 / major_axis**2))
        features['eccentricity'] = eccentricity
    else:
        features['eccentricity'] = 0

    # Яркость внутри объекта
    mask = np.zeros(image_gray.shape, dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    mean_intensity = cv2.mean(image_gray, mask=mask)[0]
    features['mean_intensity'] = mean_intensity

    return features

# Функция классификации
def classify_anomaly(features):
    print(features['aspect_ratio'], features['compactness'], features['area'], features['eccentricity'], features['mean_intensity'])
    if features['aspect_ratio'] > 0.9 and features['compactness'] > 15 and features['eccentricity'] > 0.6:
        return 'Scratch', (255, 0, 0)  # Красный
    elif features['mean_intensity'] > 140 and features['eccentricity'] < 0.5 and features['compactness'] < 10:
        return 'Chip', (0, 0, 255)  # Синий
    elif features['eccentricity'] < 0.9:
        return 'Bevel', (0, 255, 0)  # Зеленый
    elif features['mean_intensity'] < 100 and features['compactness'] < 10 and features['aspect_ratio'] < 1:
        return 'Dent', (0, 165, 255)  # Оранжевый
    elif features['mean_intensity'] < 300 and features['compactness'] < 7 and features['eccentricity'] < 1:
        return 'Scuff', (255, 0, 255)  # Фиолетовый
    else:
        return 'Unknown', (0, 255, 255)  # Желтый для неизвестной

# Функция для увеличения эллипса
def increase_ellipse(ellipse, scale_factor=1.2):
    center, axes, angle = ellipse
    major_axis, minor_axis = axes

    # Увеличиваем оси
    new_major_axis = major_axis * scale_factor
    new_minor_axis = minor_axis * scale_factor

    # Возвращаем увеличенный эллипс
    return (center, (new_major_axis, new_minor_axis), angle)

def compare_neighboring_regions(image, region_size=0.05, threshold=8, contour_area_threshold=300): 
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
    masked_image = cv2.bitwise_and(result_image, result_image, mask=cv2.bitwise_not(anomaly_map))
    
    # Применяем алгоритм Кэнни для поиска краев в аномальных областях
    edges = cv2.Canny(image, 100, 200)  # Алгоритм Кэнни для поиска краев

    # Поиск контуров на изображении с краями
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Рисуем контуры, пропуская слишком большие (по площади) прямоугольные контуры
    for contour in contours:
        # Вычисляем площадь контура
        area = cv2.contourArea(contour)
        cv2.drawContours(result_image, [contour], -1, (255, 0, 0), 2) 
        if area > contour_area_threshold and area < 10000:  # Фильтруем контуры по площади
            print(area)
            cv2.drawContours(real_anomaly_map, [contour], -1, (255, 255, 255), 2)  

    # cv2.imshow('result_image', result_image)
    # cv2.waitKey(0)
    masked_image = cv2.bitwise_and(result_image, result_image, mask=cv2.bitwise_not(anomaly_map))

    return result_image, anomaly_map
    
    

def detect_and_save_anomalies(input_image, reference_image, output_folder, threshold=80, region_size=0.05, min_anomaly_size=100, dilate_iter=5):
    """
    Основной алгоритм обнаружения аномалий и их сохранения в виде отдельных изображений.
    """
    start_con = time.time()

    # Преобразование в градации серого
    input_gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    reference_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
    
    # Вычисление разницы между изображениями
    diff = cv2.absdiff(input_gray, reference_gray)
    # cv2.imshow('diff', diff)
    # cv2.waitKey(0)
    # Применение порога для выделения аномалий
    _, anomalies = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    anomalies_contours, _ = cv2.findContours(anomalies, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest =  max(anomalies_contours, key=cv2.contourArea)
    cv2.drawContours(anomalies, [largest], -1, (0, 0, 0), 16)
    # cv2.drawContours(diff, [largest], -1, (0, 0, 0), 16)

    # cv2.imshow('diff', diff)
    # cv2.waitKey(0)
    # cv2.imshow('anomalies', anomalies)
    # cv2.waitKey(0)
    # Обнаружение аномалий на основе соседних областей
    # _, region_anomalies = compare_neighboring_regions(input_gray, region_size=region_size)
    # cv2.imshow('region_anomalies', region_anomalies)
    # cv2.waitKey(0)
    # Комбинирование результатов
    # combined_anomalies = cv2.bitwise_and(anomalies, anomalies)
    # cv2.imshow('combined_anomalies', combined_anomalies)
    # cv2.waitKey(0)

    # Увеличение аномалий для объединения близко расположенных, iterations=dilate_iter
    kernel = np.ones((3, 3), np.uint8)
    combined_anomalies = cv2.dilate(anomalies, kernel)
    
    # Поиск контуров
    
    contours, _ = cv2.findContours(combined_anomalies, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    finish_con = time.time()

        # Вывод времени выполнения
    res_msec_con = (finish_con - start_con) 
    print('Время работы на одном кадре в миллисекундах: ', res_msec_con)
    # cv2.imshow('combined_anomalies', combined_anomalies)
    # cv2.waitKey(0)
    # Рисование контуров и сохранение аномалий
    
    anomaly_index = 0
    print(len(contours))
    for contour in contours:
        output_image = input_image.copy()
        output_gray = input_gray.copy()
        if cv2.contourArea(contour) < min_anomaly_size:
            continue
        
        if len(contour) < 5:
            continue

        # Извлечение признаков
        area = cv2.contourArea(contour)
        print(area)
        if area > 18000:
            continue
        ellipse = cv2.fitEllipse(contour)
        ellipse = increase_ellipse(ellipse, scale_factor=1.3)
        # Рассчитываем характеристики
        features = calculate_features(contour, output_gray)

        # Классифицируем аномалию
        anomaly_type, colour = classify_anomaly(features)

        # Добавление метки на изображение
        x, y, w, h = cv2.boundingRect(contour)
        cv2.putText(output_image, anomaly_type, (x, y - 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, colour, 2)
        cv2.ellipse(output_image, ellipse, colour, 2)
        
        # Выделяем аномальную область и сохраняем ее       
        anomaly_filename = f"{output_folder}/anomaly_{anomaly_index}.png"
        cv2.imwrite(anomaly_filename, output_image)

        anomaly_index += 1

    return output_image

# Загрузка изображений
input_image = cv2.imread("./ourdets/blue/refAnddef/def.jpg")
reference_image = cv2.imread("./ourdets/blue/refAnddef/1/ref.jpg")
output_folder = "./output_anomalies/"

start = time.time()
# Обнаружение аномалий и сохранение результатов
output_image = detect_and_save_anomalies(input_image, reference_image, output_folder)
finish = time.time()

# Вывод времени выполнения
res_msec = (finish - start) * 1000
print('Время работы в миллисекундах: ', res_msec)
# Сохранение итогового изображения
cv2.imwrite("output_image_with_circles.png", output_image)
