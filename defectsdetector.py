import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import time

# Функция для расчета характеристик
def calculate_features(contour, image_gray):
    features = {}

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
    aspect_ratio = w / h if h > 0 else 0
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

    # яркость внутри объекта
    mask = np.zeros_like(image_gray, dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
    mean_intensity = cv2.mean(image_gray, mask=mask)[0]
    features['mean_intensity'] = mean_intensity

    return features

import pandas as pd

def classify_anomaly(features, contour, image_gray):
    x, y, w, h = cv2.boundingRect(contour)
    contour_local = contour - np.array([[x, y]])

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask, [contour_local], -1, 255, -1)

    cropped_image = image_gray[y:y+h, x:x+w]
    black_pixel_ratio = np.sum(cropped_image == 0) / np.sum(cropped_image)

    print(features['compactness'], features['eccentricity'], features['aspect_ratio'], features['mean_intensity'])
    if black_pixel_ratio > 0.01:
        return None, None  # Это отверстие, игнорируем
    print(black_pixel_ratio)
    if features['aspect_ratio'] < 7 and features['compactness'] > 3 and features['eccentricity'] > 0.6 and features['mean_intensity'] > 170:
        return 'Scratch', (255, 0, 0)  
    elif features['aspect_ratio'] > 0.9 and features['compactness'] > 15 and features['eccentricity'] > 0.6:
        return 'Scratch', (255, 0, 0)  
    elif features['mean_intensity'] > 140 and features['eccentricity'] < 0.5 and features['compactness'] < 10:
        return 'Chip', (0, 0, 255)  
    elif features['eccentricity'] < 0.9:
        return 'Bevel', (0, 255, 0)  
    elif features['mean_intensity'] < 100 and features['compactness'] < 10 and features['aspect_ratio'] < 1:
        return 'Dent', (0, 165, 255) 
    elif features['mean_intensity'] < 300 and features['compactness'] < 7 and features['eccentricity'] < 1:
        return 'Scuff', (255, 0, 255)  
    else:
        return 'Unknown', (0, 255, 255) 

def increase_ellipse(ellipse, scale_factor=1.2):
    center, axes, angle = ellipse
    major_axis, minor_axis = axes

    new_major_axis = major_axis * scale_factor
    new_minor_axis = minor_axis * scale_factor

    return (center, (new_major_axis, new_minor_axis), angle)

def detect_and_highlight_anomalies(obj_img, region_size_ratio=0.05, anomaly_threshold=0.7, pixel_diff_threshold=40):
    h, w = obj_img.shape
    region_h = int(h * region_size_ratio)
    region_w = int(w * region_size_ratio)

    region_h = max(1, region_h)
    region_w = max(1, region_w)

    _, object_mask = cv2.threshold(obj_img, 1, 255, cv2.THRESH_BINARY)
    
    result_img = cv2.cvtColor(obj_img, cv2.COLOR_GRAY2BGR)
    anomalies_mask = np.zeros_like(obj_img, dtype=np.uint8)

    for y in range(0, h, region_h):
        for x in range(0, w, region_w):
            region = obj_img[y:y+region_h, x:x+region_w]
            region_mask = object_mask[y:y+region_h, x:x+region_w]
            
            if cv2.countNonZero(region_mask) == 0:
                continue
            
            surrounding_top = max(0, y - region_h)
            surrounding_bottom = min(h, y + 2 * region_h)
            surrounding_left = max(0, x - region_w)
            surrounding_right = min(w, x + 2 * region_w)

            surrounding = obj_img[surrounding_top:surrounding_bottom, surrounding_left:surrounding_right]
            surrounding_mask = object_mask[surrounding_top:surrounding_bottom, surrounding_left:surrounding_right]

            mask = np.ones_like(surrounding, dtype=bool)
            mask[y - surrounding_top:y + region_h - surrounding_top, 
                 x - surrounding_left:x + region_w - surrounding_left] = False
            surrounding_values = surrounding[mask & (surrounding_mask > 0)]  # Учитываем только объект
            
            if len(surrounding_values) == 0:
                continue

            # Вычисление средней интенсивности окружения
            surrounding_mean = np.mean(surrounding_values)
            
            
            # Разница между текущей областью и окружением
            region_diff = np.abs(region.astype(np.float32) - surrounding_mean)
            print(np.mean(region_diff))
            if np.mean(region_diff) > 20:
                # Выделение аномальных пикселей по порогу
                _, binary_diff = cv2.threshold(region_diff, pixel_diff_threshold, 255, cv2.THRESH_BINARY)
                binary_diff = binary_diff.astype(np.uint8)
                # cv2.imshow('diff', binary_diff)
                # cv2.waitKey(0)
                anomalies_mask[y:y+region_h, x:x+region_w] = cv2.bitwise_or(anomalies_mask[y:y+region_h, x:x+region_w], binary_diff)
            
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (27, 27))
    processed_mask = cv2.morphologyEx(anomalies_mask, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(processed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) > 5:  # Игнорируем слишком маленькие области
            cv2.drawContours(result_img, [contour], -1, (0, 0, 255), 2)  
    _, mask = cv2.threshold(obj_img, 1, 255, cv2.THRESH_BINARY)
    anomalies_mask = cv2.bitwise_and(anomalies_mask, anomalies_mask, mask=mask)
    # cv2.imshow('hjasd', result)
    # cv2.waitKey(0)
    return result_img, anomalies_mask

def detect_differ(obj_img, ref_img, region_size_ratio=0.05, anomaly_threshold=0.5, pixel_diff_threshold=50):

    if obj_img.shape != ref_img.shape:
        raise ValueError("Размеры изображения объекта и эталонного изображения должны совпадать")

    h, w = obj_img.shape
    region_h = int(h * region_size_ratio)
    region_w = int(w * region_size_ratio)

    region_h = max(1, region_h)
    region_w = max(1, region_w)

    result_img = cv2.cvtColor(obj_img, cv2.COLOR_GRAY2BGR) 
    anomalies_mask = np.zeros_like(obj_img, dtype=np.uint8)

    for y in range(0, h, region_h):
        for x in range(0, w, region_w):         
            obj_region = obj_img[y:y+region_h, x:x+region_w]
            ref_region = ref_img[y:y+region_h, x:x+region_w]

            if obj_region.shape == ref_region.shape and obj_region.size > 0:
                score, _ = ssim(obj_region, ref_region, full=True)
                if score < anomaly_threshold:
                    diff = cv2.absdiff(obj_region, ref_region)
                    _, binary_diff = cv2.threshold(diff, pixel_diff_threshold, 255, cv2.THRESH_BINARY)
                    anomalies_mask[y:y+region_h, x:x+region_w] = binary_diff

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (27, 27))
    anomalies_mask2 = cv2.morphologyEx(anomalies_mask, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(anomalies_mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) > 5:  # Игнорируем слишком маленькие области
            cv2.drawContours(result_img, [contour], -1, (0, 0, 255), 2)  

    return result_img, anomalies_mask

def detect_and_save_anomalies(input_image, reference_image, output_folder, threshold=50, region_size=0.05, min_anomaly_size=20, dilate_iter=5):
    """
    Основной алгоритм обнаружения аномалий и их сохранения в виде отдельных изображений.
    """  
    input_gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    reference_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
    reference_gray2 = cv2.GaussianBlur(reference_gray, (5, 5), 0)
   
    input_gray2 = cv2.GaussianBlur(input_gray, (7, 7), 0)
    
    highlighted_result, detailed_mask = detect_and_highlight_anomalies(input_gray)
    _, mask_ref = detect_and_highlight_anomalies(reference_gray, pixel_diff_threshold=20)
    # cv2.imshow('input_image', input_image)
    # cv2.imshow('reference_image', reference_image)
    # cv2.imshow('detailed_mask', detailed_mask)
    # cv2.imshow('mask_ref', mask_ref)
    # cv2.waitKey(0)
    mask_ref = cv2.bitwise_not(mask_ref)
    # cv2.imshow('mask_ref', mask_ref)
    detailed_mask_combined = cv2.bitwise_and(detailed_mask, mask_ref)
    # cv2.imshow('detailed_mask_combined', detailed_mask_combined)
    
    _, differ = detect_differ(input_gray2, reference_gray2, region_size_ratio=0.1)
    # cv2.imshow('differ', differ)
    # cv2.waitKey(0)
    combined_anomalies = cv2.bitwise_or(detailed_mask_combined, differ)
    contours, _ = cv2.findContours(combined_anomalies, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.imshow('combined_anomalies', combined_anomalies)
    cv2.waitKey(0)
    test = input_image.copy()
    anomaly_index = 0

    for contour in contours:
        output_image = input_image.copy()
        if cv2.contourArea(contour) < min_anomaly_size:
            continue
        
        if len(contour) < 5:
            continue

        features = calculate_features(contour, input_gray)

        anomaly_type, colour = classify_anomaly(features, contour, input_gray)
        if anomaly_type is None:
            continue  # Пропускаем отверстие
        cv2.drawContours(test, [contour], -1, (0, 0, 255), 2)
        x, y, w, h = cv2.boundingRect(contour)
        ellipse = cv2.fitEllipse(contour)
        ellipse = increase_ellipse(ellipse, scale_factor=1.5)

        if (
            np.all(np.isfinite(ellipse[0])) and
            np.all(np.isfinite(ellipse[1])) and
            ellipse[1][0] > 0 and ellipse[1][1] > 0
        ):
            cv2.putText(output_image, anomaly_type, (x, y - 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, colour, 2)
            cv2.ellipse(output_image, ellipse, colour, 2)
           
            anomaly_filename = f"{output_folder}/anomaly_{anomaly_index}.png"
            print(anomaly_filename)
            cv2.imwrite(anomaly_filename, output_image)
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
