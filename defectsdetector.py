import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import time
from skimage import io, filters, morphology

# расчет характеристик для выделения аномалий
def calculate_features(contour, image_gray):
    features = {}

    # площадь и периметр
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    features['area'] = area
    features['perimeter'] = perimeter

    # компактность (1 - идеальная окружность)
    if area > 0:
        compactness = (perimeter ** 2) / (4 * np.pi * area)
        features['compactness'] = compactness
    else:
        features['compactness'] = 0

    # ограничивающий прямоугольник
    x, y, w, h = cv2.boundingRect(contour)
    # соотношение сторон
    aspect_ratio = w / h
    features['aspect_ratio'] = aspect_ratio

    # эллипс (если достаточно точек)
    if len(contour) >= 5:
        ellipse = cv2.fitEllipse(contour)
        (center, axes, angle) = ellipse
        major_axis = max(axes)
        minor_axis = min(axes)
        # более вытянутая, например, царапина 
        eccentricity = np.sqrt(1 - (minor_axis**2 / major_axis**2))
        features['eccentricity'] = eccentricity
    else:
        features['eccentricity'] = 0

    # яркость внутри объекта
    mask = np.zeros(image_gray.shape, dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    mean_intensity = cv2.mean(image_gray, mask=mask)[0]
    features['mean_intensity'] = mean_intensity

    return features

# классификации
def classify_anomaly(features):
    print(features['aspect_ratio'], features['compactness'], features['area'], features['eccentricity'], features['mean_intensity'])
    if features['aspect_ratio'] > 0.9 and features['compactness'] > 15 and features['eccentricity'] > 0.6:
        return 'Scratch', (255, 0, 0)  # царапина - синий
    elif features['mean_intensity'] > 140 and features['eccentricity'] < 0.5 and features['compactness'] < 10:
        return 'Chip', (0, 0, 255)  # скол - красный
    elif features['eccentricity'] < 0.9:
        return 'Bevel', (0, 255, 0)  # скос - зеленый
    elif features['mean_intensity'] < 100 and features['compactness'] < 10 and features['aspect_ratio'] < 1:
        return 'Dent', (0, 165, 255)  # вмятина - оранжевый
    elif features['mean_intensity'] < 300 and features['compactness'] < 7 and features['eccentricity'] < 1:
        return 'Scuff', (255, 0, 255)  # потертость - розовый
    else:
        return 'Unknown', (0, 255, 255)  # неизвестная аномалия - желтный

# увеличение ограничивающего эллипса
def increase_ellipse(ellipse, scale_factor=1.2):
    center, axes, angle = ellipse
    major_axis, minor_axis = axes

    # увеличиваем оси
    new_major_axis = major_axis * scale_factor
    new_minor_axis = minor_axis * scale_factor

    return (center, (new_major_axis, new_minor_axis), angle)

def detect_and_highlight_anomalies(obj_img, ref_img, region_size_ratio=0.05, anomaly_threshold=0.7, pixel_diff_threshold=30):
    if obj_img.shape != ref_img.shape:
        raise ValueError("Размеры изображения объекта и эталонного изображения должны совпадать")

    h, w = obj_img.shape
    region_h = int(h * region_size_ratio)
    region_w = int(w * region_size_ratio)

    region_h = max(1, region_h)
    region_w = max(1, region_w)

    _, object_mask = cv2.threshold(obj_img, 1, 255, cv2.THRESH_BINARY)
    # cv2.imshow('obj_img', obj_img)
    # cv2.imshow('massssk', object_mask)
    # cv2.waitKey(0)
    result_img = cv2.cvtColor(obj_img, cv2.COLOR_GRAY2BGR)  # Для цветного выделения
    anomalies_mask = np.zeros_like(obj_img, dtype=np.uint8)

    # проход по сетке областей
    for y in range(0, h, region_h):
        for x in range(0, w, region_w):
            # текущая область
            region = obj_img[y:y+region_h, x:x+region_w]
            region_mask = object_mask[y:y+region_h, x:x+region_w]

            # если область вне объекта, пропускаем
            if cv2.countNonZero(region_mask) == 1:
                continue

            # координаты окружения
            surrounding_top = max(0, y - region_h)
            surrounding_bottom = min(h, y + 2 * region_h)
            surrounding_left = max(0, x - region_w)
            surrounding_right = min(w, x + 2 * region_w)

            # окружение (с учётом маски объекта)
            surrounding = obj_img[surrounding_top:surrounding_bottom, surrounding_left:surrounding_right]
            surrounding_mask = object_mask[surrounding_top:surrounding_bottom, surrounding_left:surrounding_right]

            # исключение текущей области из окружения
            mask = np.ones_like(surrounding, dtype=bool)
            mask[y - surrounding_top:y + region_h - surrounding_top, 
                 x - surrounding_left:x + region_w - surrounding_left] = False
            surrounding_values = surrounding[mask & (surrounding_mask > 0)]  # только объект
            
            # если окружение пустое, пропускаем
            if len(surrounding_values) == 0:
                continue

            # средняя интенсивность окружения
            surrounding_mean = np.mean(surrounding_values)

            # разница между текущей областью и окружением
            region_diff = np.abs(region.astype(np.float32) - surrounding_mean)

            # выделение аномальных пикселей по порогу
            _, binary_diff = cv2.threshold(region_diff, pixel_diff_threshold, 255, cv2.THRESH_BINARY)
            binary_diff = binary_diff.astype(np.uint8)

            # добавление к общей маске
            anomalies_mask[y:y+region_h, x:x+region_w] = cv2.bitwise_or(anomalies_mask[y:y+region_h, x:x+region_w], binary_diff)
            
    # морфологическая обработка маски
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (27, 27))
    processed_mask = cv2.morphologyEx(anomalies_mask, cv2.MORPH_CLOSE, kernel)
    
    # поиск контуров аномалий
    contours, _ = cv2.findContours(processed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # contours2, _ = cv2.findContours(object_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # largest_contour = max(contours2, key=cv2.contourArea)
    
    # отрисовка контуров на изображении
    for contour in contours:
        if cv2.contourArea(contour) > 5:  # слишком маленькие области
            cv2.drawContours(result_img, [contour], -1, (0, 0, 255), 2)  
    _, mask = cv2.threshold(obj_img, 1, 255, cv2.THRESH_BINARY)
    anomalies_mask = cv2.bitwise_and(anomalies_mask, anomalies_mask, mask=mask)
    # cv2.imshow('hjasd', result)
    # cv2.waitKey(0)
    return result_img, anomalies_mask

def detect_differ(obj_img, ref_img, region_size_ratio=0.05, anomaly_threshold=0.7, pixel_diff_threshold=30):
    if obj_img.shape != ref_img.shape:
        raise ValueError("Размеры изображения объекта и эталонного изображения должны совпадать")

    h, w = obj_img.shape
    region_h = int(h * region_size_ratio)
    region_w = int(w * region_size_ratio)

    region_h = max(1, region_h)
    region_w = max(1, region_w)

    result_img = cv2.cvtColor(obj_img, cv2.COLOR_GRAY2BGR)  # Для цветного выделения
    anomalies_mask = np.zeros_like(obj_img, dtype=np.uint8)

    # проход по сетке областей
    for y in range(0, h, region_h):
        for x in range(0, w, region_w):         
            # текущая область
            obj_region = obj_img[y:y+region_h, x:x+region_w]
            ref_region = ref_img[y:y+region_h, x:x+region_w]

            # сравнение областей
            if obj_region.shape == ref_region.shape and obj_region.size > 0:
                # среднее структурное сходство (яркость, контрастность, текстура)
                score, _ = ssim(obj_region, ref_region, full=True)
                if score < anomaly_threshold:
                    # разница
                    diff = cv2.absdiff(obj_region, ref_region)

                    # выделение аномальных пикселей по порогу
                    _, binary_diff = cv2.threshold(diff, pixel_diff_threshold, 255, cv2.THRESH_BINARY)

                    # добавление к общей маске
                    anomalies_mask[y:y+region_h, x:x+region_w] = binary_diff

    # морфологическая обработка маски
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (27, 27))
    processed_mask = cv2.morphologyEx(anomalies_mask, cv2.MORPH_CLOSE, kernel)
    
    # поиск контуров аномалий
    contours, _ = cv2.findContours(processed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # отрисовка контуров на изображении
    for contour in contours:
        if cv2.contourArea(contour) > 5:  # слишком маленькие области
            cv2.drawContours(result_img, [contour], -1, (0, 0, 255), 2)  

    return result_img, anomalies_mask

# объединение ближайших контуров
def combine_nearby_contours(contours, max_distance=10, max_nearby_contours=5):
    '''
    объединяет ближайшие контуры, если они находятся в пределах заданного расстояния
    '''
    combined_contours = []
    used = [False] * len(contours)  # массив для отслеживания уже объединённых контуров

    for i in range(len(contours)):
        if used[i]:
            continue

        combined_contour = contours[i]
        used[i] = True

        # центроид текущего контура
        moments = cv2.moments(combined_contour)
        if moments['m00'] == 0:
            continue 
        
        # пропуск контуро с нулевой площадью
        cX = int(moments['m10'] / moments['m00'])
        cY = int(moments['m01'] / moments['m00'])

        # если контуры близки - объединяем
        for j in range(i + 1, len(contours)):
            if used[j]:
                continue

            # центроид второго контура
            moments2 = cv2.moments(contours[j])
            if moments2['m00'] == 0:
                continue 
            cX2 = int(moments2['m10'] / moments2['m00'])
            cY2 = int(moments2['m01'] / moments2['m00'])

            # расстояние между центроидами
            distance = np.sqrt((cX - cX2) ** 2 + (cY - cY2) ** 2)

            # если расстояние меньше максимального порога, объединяем контуры
            if distance < max_distance:
                combined_contour = np.concatenate((combined_contour, contours[j]), axis=0)
                used[j] = True  # отметка второго контура как использованного

        combined_contours.append(combined_contour)

    return combined_contours

def detect_and_save_anomalies(input_image, reference_image, output_folder, threshold=50, region_size=0.05, min_anomaly_size=10, dilate_iter=5):
    '''
    обнаружение аномалий и их сохранения в виде отдельных изображений
    '''
    start_con = time.time()

    input_gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    reference_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)

    # cv2.imshow('img', input_gray)
    # cv2.imshow('reference_gray', reference_gray)
    # cv2.waitKey(0)
    reference_gray = cv2.GaussianBlur(reference_gray, (5, 5), 0)
    # сравнение по соседним областям (не отредачено, должно подаваться только изображение детали с дефектом)
    highlighted_result, detailed_mask = detect_and_highlight_anomalies(input_gray, reference_gray)

    # сравнение по областям двух избражений
    _, differ = detect_differ(input_gray, reference_gray)

    # diff = cv2.absdiff(input_gray, reference_gray)
    # _, anomalies = cv2.threshold(diff, 1000, 255, cv2.THRESH_BINARY)
    cv2.imshow('differ', differ)
    cv2.imshow('detailed_mask', detailed_mask)
    # объединение масок
    combined_anomalies = cv2.bitwise_and(detailed_mask, differ)
    # kernel = np.ones((3, 3), np.uint8)
    # combined_anomalies = cv2.dilate(combined_anomalies, kernel)
    cv2.imshow('combined_anomalies', combined_anomalies)
    cv2.waitKey(0)
    # поиск контуров
    contours, _ = cv2.findContours(combined_anomalies, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # объединение ближайших контуров
    contours = combine_nearby_contours(contours)
    test = input_image.copy()
    anomaly_index = 0
    for contour in contours:
        output_image = input_image.copy()
        output_gray = input_gray.copy()
        if cv2.contourArea(contour) < min_anomaly_size:
            continue
        
        if len(contour) < 5:
            continue

        area = cv2.contourArea(contour)
        print(area)
        # if area > 18000:
        #     continue
        cv2.drawContours(test, [contour], -1, (255, 255, 255), 2) 
        ellipse = cv2.fitEllipse(contour)
        ellipse = increase_ellipse(ellipse, scale_factor=1.5)

        # характеристики
        features = calculate_features(contour, output_gray)

        # классификация аномалии
        anomaly_type, colour = classify_anomaly(features)

        x, y, w, h = cv2.boundingRect(contour)
        if (
        np.all(np.isfinite(ellipse[0])) and  
        np.all(np.isfinite(ellipse[1])) and  
        ellipse[1][0] > 0 and ellipse[1][1] > 0  
        ):
            cv2.putText(output_image, anomaly_type, (x, y - 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, colour, 2)
            cv2.ellipse(output_image, ellipse, colour, 2)
            anomaly_filename = f"{output_folder}/anomaly_{anomaly_index}.png"
            # cv2.imwrite(anomaly_filename, output_image)
            print(anomaly_filename)
            anomaly_index += 1
        else:
            print(f"Некорректный эллипс: {ellipse}")
    cv2.imshow('test', test)
    cv2.waitKey(0)
    return output_image

input_image = cv2.imread("./ourdets/blue/refAnddef/def.jpg")
reference_image = cv2.imread("./ourdets/blue/refAnddef/1/ref.jpg")
output_folder = "./output_anomalies/"

start = time.time()
output_image = detect_and_save_anomalies(input_image, reference_image, output_folder)
finish = time.time()

res_msec = (finish - start) * 1000
print('Время работы в миллисекундах: ', res_msec)
cv2.imwrite("output_image_with_circles.png", output_image)
