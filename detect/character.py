import cv2
import numpy as np
import time

from features import calculate_features, save_features_to_excel
# from classify_anomalies import classify_anomaly
from detect_anomalies import detect_and_highlight_anomalies, detect_differ
from contours_connection import connect_contours, remove_nested_contours, merge_overlapping_contours
from det_aligning import align
from backremoveCV import remover


def resize_and_align_reference(input_image, reference_image):
    """
    "Вырезает" деталь из reference_image по наибольшему контуру, создает черный фон по размерам input_image и вставляет вырезанную деталь в соответствующее место.
    :param input_image: Входное изображение, к размеру которого нужно привести reference_image.
    :param reference_image: Эталонное изображение, которое нужно масштабировать и привести к размеру input_image.
    :return: Преобразованное reference_image.
    """
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

def detect_and_save_anomalies(input_image, reference_image, output_folder, threshold=50, region_size=0.05, min_anomaly_size=20, dilate_iter=5):
    """
    Основной алгоритм обнаружения аномалий и их сохранения в виде отдельных изображений.
    """  
    # Приведение reference_image к input_image
    reference_image = resize_and_align_reference(input_image, reference_image)
    # cv2.imshow('input_image', input_image)
    # cv2.imshow('reference_image', reference_image)
    # cv2.waitKey(0)
    input_gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    contours_max, _ = cv2.findContours(input_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
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

        reference_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
        reference_gray2 = cv2.GaussianBlur(reference_gray, (7, 7), 0)
        input_gray2 = cv2.GaussianBlur(input_gray, (7, 7), 0) 
    
        cv2.imwrite('./input_gray.jpg', input_gray)
        _, detailed_mask = detect_and_highlight_anomalies(input_gray, pixel_diff_threshold=60)
        kernel = np.ones((2, 2), np.uint8)
        detailed_mask = cv2.dilate(detailed_mask, kernel, iterations=1)
        cv2.imwrite('./detailed_mask.jpg', detailed_mask)
     
        _, mask_ref = detect_and_highlight_anomalies(reference_gray, pixel_diff_threshold=60)
        cv2.imwrite('./mask_ref.jpg', mask_ref)
        mask_ref = cv2.bitwise_not(mask_ref)

        cv2.imwrite('./bitwise_not_mask_ref.jpg', mask_ref)

        
        detailed_mask_combined = cv2.bitwise_and(detailed_mask, mask_ref)
        cv2.imwrite('./detailed_mask_combined.jpg', detailed_mask_combined)
        
        _, differ = detect_differ(input_gray2, reference_gray2, region_size_ratio=0.05)
        cv2.imwrite('./differ.jpg', differ)
        combined_anomalies = cv2.bitwise_or(detailed_mask_combined, differ)
        combined_anomalies = cv2.bitwise_and(combined_anomalies, mask_ref)
        
        cv2.imwrite('./combined_anomalies.jpg', combined_anomalies)
        contours, _ = cv2.findContours(combined_anomalies, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
       
        contours_new = []
         
        contours = connect_contours(contours, input_gray, max_area, max_perimeter, max_centroid, min_dist=5)
        contours = connect_contours(contours, input_gray, max_area, max_perimeter, max_centroid, min_dist=5)
       
        all_cont_ing = input_image.copy()
        for contour in contours:
            
            cv2.drawContours(all_cont_ing, [contour], -1, (0, 0, 255), 2)
            if cv2.contourArea(contour) < min_anomaly_size:
                continue
            
            features_remove = calculate_features(contour, input_gray, max_area, max_perimeter, max_centroid)
            # save_features_to_excel(features_remove, './spam/anomalies.xlsx')
            # features_remove['relative_area'] < 0.05 and 
            
            if features_remove['relative_area'] < 0.0186:
                print(features_remove['relative_area'], features_remove['mean_intensity'])
                contours_new.append(contour)
        cv2.imwrite('./all_cont_ing.jpg', all_cont_ing)
        
        contours = contours_new
        contours = merge_overlapping_contours(contours, input_gray, max_area, max_perimeter, max_centroid)
        contours = connect_contours(contours, input_gray, max_area, max_perimeter, max_centroid, min_dist=5)
        contours = remove_nested_contours(contours, input_gray, max_area, max_perimeter, max_centroid)
        test = input_image.copy()
        anomaly_index = 0
        for contour in contours:
            if cv2.contourArea(contour) < 15:
                continue
            output_image = input_image.copy()
            
            features = calculate_features(contour, input_gray, max_area, max_perimeter, max_centroid)

            anomaly_type, colour = None, None
            if anomaly_type is None:
                colour = (255,255,0)
                anomaly_type = 'Unknown' 
            
            x, y, w, h = cv2.boundingRect(contour)
          
            cv2.drawContours(test, [contour], -1, (0, 0, 255), 2)
            cv2.rectangle(output_image, (x, y), (x + w, y + h), colour, 2) 
            cv2.rectangle(test, (x, y), (x + w, y + h), colour, 2) 
            anomaly_filename = f"{output_folder}/anomaly_{anomaly_index}.png"
            cv2.imwrite(anomaly_filename, output_image)

            anomaly_index += 1
         
        cv2.imwrite('./test.jpg', test)


def remove_back_and_detect(img_path, reference_path, output_folder):
    
    input_image = remover(cv2.imread(img_path))
    cv2.imwrite('./input_image1.jpg', input_image)

    reference_image = remover(cv2.imread(reference_path))
    cv2.imwrite('./reference_image.jpg', reference_image)

    start = time.time()
    detect_and_save_anomalies(input_image, reference_image, output_folder)
    finish = time.time()

    res_msec = (finish - start) * 1000
    print('Время работы в миллисекундах: ', res_msec)
   

img_path = r'./def52.jpg'
reference_path = r'./ref5.jpg'
output_folder = r'./output_folder'

remove_back_and_detect(img_path, reference_path, output_folder)
