import cv2
import numpy as np
import time
import os
import sys
import pandas as pd

# from classify_anomalies import classify_anomaly

sys.path.append(os.path.abspath('./detect'))
from detect.features import calculate_features, save_features_to_excel
from detect.detect_anomalies import detect_and_highlight_anomalies, detect_differ
from detect.contours_connection import connect_contours, remove_nested_contours, merge_overlapping_contours

sys.path.append(os.path.abspath('./preprocess'))
from preprocess.backremoveCV import remover
from preprocess.prep_metal import preprocess_metal_image
from preprocess.aligning import align

# sys.path.append(os.path.abspath('./threed'))
# from threed.rotate import find_temp

def resize_and_align_reference(input_image, reference_image):
    '''
    "Вырезает" деталь из reference_image по наибольшему контуру, создает черный фон по размерам input_image и вставляет вырезанную деталь в соответствующее место.
    :param input_image: Входное изображение, к размеру которого нужно привести reference_image.
    :param reference_image: Эталонное изображение, которое нужно масштабировать и привести к размеру input_image.
    :return: Преобразованное reference_image.
    '''
    # Преобразуем изображения в оттенки серого
    input_gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    reference_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)

    # Находим наибольшие контуры на обоих изображениях
    contours_input, _ = cv2.findContours(input_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_ref, _ = cv2.findContours(reference_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_contour_input = max(contours_input, key=cv2.contourArea) if contours_input else None
    max_contour_ref = max(contours_ref, key=cv2.contourArea) if contours_ref else None

    if max_contour_input is None or max_contour_ref is None:
        raise ValueError("Не удалось найти контуры на одном из изображений.")

    # Вычисляем прямоугольники, охватывающие наибольшие контуры
    x_input, y_input, w_input, h_input = cv2.boundingRect(max_contour_input)
    x_ref, y_ref, w_ref, h_ref = cv2.boundingRect(max_contour_ref)
    
    if w_input <= 0 or h_input <= 0:
        raise ValueError("Некорректные размеры для масштабирования изображения.")

    # Вырезаем деталь из reference_image
    detail = reference_image[y_ref:y_ref+h_ref, x_ref:x_ref+w_ref]

    # Масштабируем деталь к размерам прямоугольника на input_image
    scaled_detail = cv2.resize(detail, (w_input, h_input), interpolation=cv2.INTER_LINEAR)

    # Создаем черный фон по размерам input_image
    input_h, input_w = input_image.shape[:2]
    aligned_reference = np.zeros((input_h, input_w, 3), dtype=np.uint8)

    # Вставляем масштабированную деталь в центр соответствующего прямоугольника
    aligned_reference[y_input:y_input+h_input, x_input:x_input+w_input] = scaled_detail

    return aligned_reference

def detect_and_save_anomalies(img_path, input_image, reference_image, output_folder, mode, count_def, region_size=0.05, min_anomaly_size=20):
    '''
    Основной алгоритм обнаружения аномалий и их сохранения в виде отдельных изображений.
    '''  
    if mode not in [0, 1]:
        raise ValueError("Некорректный режим работы. Допустимые значения: 0 или 1.")

    # Приведение reference_image к input_image
    reference_image = resize_and_align_reference(input_image, reference_image)

    input_gray =  preprocess_metal_image(input_image)
    contours_max, _ = cv2.findContours(input_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours_max:
        raise ValueError("Контуры не найдены на входном изображении.")

    # наибольший контур по площади
    max_contour = max(contours_max, key=cv2.contourArea) if contours_max else None
    if max_contour is not None:
        # площадь и периметр
        max_area = cv2.contourArea(max_contour)
        max_perimeter = cv2.arcLength(max_contour, True)

        # вычисление центра масс
        M = cv2.moments(max_contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            max_centroid = (cx, cy)
        else:
            max_centroid = (0, 0)  # если момент равен нулю (неправильный контур)

        reference_gray = preprocess_metal_image(reference_image)
        reference_gray2 = cv2.GaussianBlur(reference_gray, (7, 7), 0)
        input_gray2 = cv2.GaussianBlur(input_gray, (7, 7), 0) 
        if mode==0:
            threshold=60
        else:
            threshold=75
        _, detailed_mask = detect_and_highlight_anomalies(input_gray, pixel_diff_threshold=threshold)
        kernel = np.ones((2, 2), np.uint8)
        detailed_mask = cv2.dilate(detailed_mask, kernel, iterations=1)
     
        _, mask_ref = detect_and_highlight_anomalies(reference_gray, pixel_diff_threshold=threshold)
        mask_ref = cv2.bitwise_not(mask_ref)
        
        detailed_mask_combined = cv2.bitwise_and(detailed_mask, mask_ref)
        
        _, differ = detect_differ(input_gray2, reference_gray2, region_size_ratio=0.05)
        combined_anomalies = cv2.bitwise_and(cv2.bitwise_or(detailed_mask_combined, differ), mask_ref)
        
        contours, _ = cv2.findContours(combined_anomalies, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if mode==0:
            contours = connect_contours(contours, input_gray, max_area, max_perimeter, max_centroid, min_dist=5)

            contours_new = []
            for contour in contours:
                if cv2.contourArea(contour) < min_anomaly_size:
                    continue
            
            contours = merge_overlapping_contours(contours, input_gray, max_area, max_perimeter, max_centroid)
            contours = remove_nested_contours(contours, input_gray, max_area, max_perimeter, max_centroid)
        if mode==1:
            contours_new = []
            for contour in contours:
                features_remove = calculate_features(contour, input_gray, max_area, max_perimeter, max_centroid)                
                if features_remove['relative_area'] < 0.01:
                    contours_new.append(contour)
            contours = contours_new
        
        if not contours:
            raise ValueError("Ошибка: контуры не найдены или пусты")
        
        anomaly_index = 0  # Индекс аномалии
        for contour in contours:
            if cv2.contourArea(contour) < 15:
                continue
            output_image = input_image.copy()
            features = calculate_features(contour, input_gray, max_area, max_perimeter, max_centroid)
            cv2.drawContours(output_image, [contour], -1, (0, 0, 255), 1)
            anomaly_filename = f"{output_folder}/{count_def}_anomaly_{anomaly_index}.png"
            print(anomaly_index)
            cv2.imwrite(anomaly_filename, output_image)
            save_features_to_excel(features, anomaly_filename, './anomalies_features.xlsx')
            anomaly_index += 1
       

def remove_back_and_detect(img_path, reference_path, output_folder, count_def, mode=0):
    reference_image = remover(cv2.imread(reference_path))
    count_def = 0
    for file in os.listdir(img_path):
        print(file)
        image_path = os.path.join(img_path, file)
        print(image_path)
        input_image = remover(cv2.imread(image_path))
        count_def +=1
        detect_and_save_anomalies(image_path, input_image, reference_image, "./detect/output_imgs", mode, count_def)

