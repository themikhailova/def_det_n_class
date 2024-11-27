import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import time
from skimage import io, filters, morphology

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

def detect_and_highlight_anomalies(obj_img, ref_img, region_size_ratio=0.05, anomaly_threshold=0.7, pixel_diff_threshold=30):
    # Проверка размеров
    if obj_img.shape != ref_img.shape:
        raise ValueError("Размеры изображения объекта и эталонного изображения должны совпадать")

    # Размер области для анализа
    h, w = obj_img.shape
    region_h = int(h * region_size_ratio)
    region_w = int(w * region_size_ratio)

    region_h = max(1, region_h)
    region_w = max(1, region_w)

    # Создание маски объекта
    _, object_mask = cv2.threshold(obj_img, 1, 255, cv2.THRESH_BINARY)
    # cv2.imshow('obj_img', obj_img)
    # cv2.imshow('massssk', object_mask)
    # cv2.waitKey(0)
    # Результирующее изображение
    result_img = cv2.cvtColor(obj_img, cv2.COLOR_GRAY2BGR)  # Для цветного выделения
    anomalies_mask = np.zeros_like(obj_img, dtype=np.uint8)

    # Проход по сетке областей
    for y in range(0, h, region_h):
        for x in range(0, w, region_w):
            # Текущая область
            region = obj_img[y:y+region_h, x:x+region_w]
            region_mask = object_mask[y:y+region_h, x:x+region_w]

            # Если область вне объекта, пропускаем
            if cv2.countNonZero(region_mask) == 1:
                continue

            # Определяем координаты окружения
            surrounding_top = max(0, y - region_h)
            surrounding_bottom = min(h, y + 2 * region_h)
            surrounding_left = max(0, x - region_w)
            surrounding_right = min(w, x + 2 * region_w)

            # Окружение (с учётом маски объекта)
            surrounding = obj_img[surrounding_top:surrounding_bottom, surrounding_left:surrounding_right]
            surrounding_mask = object_mask[surrounding_top:surrounding_bottom, surrounding_left:surrounding_right]

            # Исключаем текущую область из окружения
            mask = np.ones_like(surrounding, dtype=bool)
            mask[y - surrounding_top:y + region_h - surrounding_top, 
                 x - surrounding_left:x + region_w - surrounding_left] = False
            surrounding_values = surrounding[mask & (surrounding_mask > 0)]  # Учитываем только объект
            
            # Если окружение пустое, пропускаем
            if len(surrounding_values) == 0:
                continue

            # Вычисление средней интенсивности окружения
            surrounding_mean = np.mean(surrounding_values)

            # Разница между текущей областью и окружением
            region_diff = np.abs(region.astype(np.float32) - surrounding_mean)

            # Выделение аномальных пикселей по порогу
            _, binary_diff = cv2.threshold(region_diff, pixel_diff_threshold, 255, cv2.THRESH_BINARY)
            binary_diff = binary_diff.astype(np.uint8)

            # Добавляем к общей маске
            anomalies_mask[y:y+region_h, x:x+region_w] = cv2.bitwise_or(anomalies_mask[y:y+region_h, x:x+region_w], binary_diff)
            
    # Морфологическая обработка маски
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (27, 27))
    processed_mask = cv2.morphologyEx(anomalies_mask, cv2.MORPH_CLOSE, kernel)
    
    # Поиск контуров аномалий
    contours, _ = cv2.findContours(processed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # contours2, _ = cv2.findContours(object_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # largest_contour = max(contours2, key=cv2.contourArea)
    
    # Отрисовка контуров на изображении
    for contour in contours:
        if cv2.contourArea(contour) > 5:  # Игнорируем слишком маленькие области
            cv2.drawContours(result_img, [contour], -1, (0, 0, 255), 2)  # Красные контуры
    _, mask = cv2.threshold(obj_img, 1, 255, cv2.THRESH_BINARY)
    anomalies_mask = cv2.bitwise_and(anomalies_mask, anomalies_mask, mask=mask)
    # cv2.imshow('hjasd', result)
    # cv2.waitKey(0)
    return result_img, anomalies_mask

def detect_differ(obj_img, ref_img, region_size_ratio=0.05, anomaly_threshold=0.7, pixel_diff_threshold=30):
    # Загрузка изображений
    # obj_img = cv2.imread(object_image, cv2.IMREAD_GRAYSCALE)
    # ref_img = cv2.imread(reference_image, cv2.IMREAD_GRAYSCALE)

    # Проверка размеров
    if obj_img.shape != ref_img.shape:
        raise ValueError("Размеры изображения объекта и эталонного изображения должны совпадать")

    # Размер области для анализа
    h, w = obj_img.shape
    region_h = int(h * region_size_ratio)
    region_w = int(w * region_size_ratio)

    region_h = max(1, region_h)
    region_w = max(1, region_w)

    # Результирующее изображение
    result_img = cv2.cvtColor(obj_img, cv2.COLOR_GRAY2BGR)  # Для цветного выделения
    anomalies_mask = np.zeros_like(obj_img, dtype=np.uint8)

    # Проход по сетке областей
    for y in range(0, h, region_h):
        for x in range(0, w, region_w):         
            # Получаем текущую область
            obj_region = obj_img[y:y+region_h, x:x+region_w]
            ref_region = ref_img[y:y+region_h, x:x+region_w]

            # Сравнение областей
            if obj_region.shape == ref_region.shape and obj_region.size > 0:
                score, _ = ssim(obj_region, ref_region, full=True)
                if score < anomaly_threshold:
                    # Рассчитываем разницу
                    diff = cv2.absdiff(obj_region, ref_region)

                    # Выделение аномальных пикселей по порогу
                    _, binary_diff = cv2.threshold(diff, pixel_diff_threshold, 255, cv2.THRESH_BINARY)

                    # Добавляем к общей маске
                    anomalies_mask[y:y+region_h, x:x+region_w] = binary_diff

    # Морфологическая обработка маски
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (27, 27))
    processed_mask = cv2.morphologyEx(anomalies_mask, cv2.MORPH_CLOSE, kernel)
    
    # Поиск контуров аномалий
    contours, _ = cv2.findContours(processed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Отрисовка контуров на изображении
    for contour in contours:
        if cv2.contourArea(contour) > 5:  # Игнорируем слишком маленькие области
            cv2.drawContours(result_img, [contour], -1, (0, 0, 255), 2)  # Красные контуры

    return result_img, anomalies_mask


def detect_and_save_anomalies(input_image, reference_image, output_folder, threshold=50, region_size=0.05, min_anomaly_size=100, dilate_iter=5):
    """
    Основной алгоритм обнаружения аномалий и их сохранения в виде отдельных изображений.
    """
    start_con = time.time()

    # Преобразование в градации серого
    input_gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    reference_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)

    # cv2.imshow('img', input_gray)
    # cv2.imshow('reference_gray', reference_gray)
    # cv2.waitKey(0)
    reference_gray = cv2.GaussianBlur(reference_gray, (5, 5), 0)
    highlighted_result, detailed_mask = detect_and_highlight_anomalies(input_gray, reference_gray)
    _, differ = detect_differ(input_gray, reference_gray)
    diff = cv2.absdiff(input_gray, reference_gray)
    _, anomalies = cv2.threshold(diff, 40, 255, cv2.THRESH_BINARY)
    cv2.imshow('differ', differ)
    cv2.imshow('detailed_mask', detailed_mask)
    combined_anomalies = cv2.bitwise_and(detailed_mask, differ)
    # kernel = np.ones((3, 3), np.uint8)
    # combined_anomalies = cv2.dilate(anomalies, kernel)
    cv2.imshow('combined_anomalies', combined_anomalies)
    cv2.waitKey(0)
    # Поиск контуров
    contours, _ = cv2.findContours(combined_anomalies, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    test = input_image.copy()
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
        # if area > 18000:
        #     continue
        cv2.drawContours(test, [contour], -1, (255, 255, 255), 2) 
        ellipse = cv2.fitEllipse(contour)
        ellipse = increase_ellipse(ellipse, scale_factor=1.3)

        # Рассчитываем характеристики
        features = calculate_features(contour, output_gray)

        # Классифицируем аномалию
        anomaly_type, colour = classify_anomaly(features)

        # Добавление метки на изображение
        x, y, w, h = cv2.boundingRect(contour)
        if (
        np.all(np.isfinite(ellipse[0])) and  # Координаты центра корректны
        np.all(np.isfinite(ellipse[1])) and  # Размеры осей корректны
        ellipse[1][0] > 0 and ellipse[1][1] > 0  # Размеры осей больше 0
        ):
            cv2.putText(output_image, anomaly_type, (x, y - 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, colour, 2)
            cv2.ellipse(output_image, ellipse, colour, 2)
            # Выделяем аномальную область и сохраняем ее       
            anomaly_filename = f"{output_folder}/anomaly_{anomaly_index}.png"
            cv2.imwrite(anomaly_filename, output_image)
            print(anomaly_filename)
            anomaly_index += 1
        else:
            print(f"Некорректный эллипс: {ellipse}")
    cv2.imshow('test', test)
    cv2.waitKey(0)
    return output_image

# Загрузка изображений
input_image = cv2.imread("./1.jpg")
reference_image = cv2.imread("./2.jpg")
output_folder = "./output_anomalies2/"

start = time.time()
# Обнаружение аномалий и сохранение результатов
output_image = detect_and_save_anomalies(input_image, reference_image, output_folder)
finish = time.time()

# Вывод времени выполнения
res_msec = (finish - start) * 1000
print('Время работы в миллисекундах: ', res_msec)
# Сохранение итогового изображения
cv2.imwrite("output_image_with_circles.png", output_image)
