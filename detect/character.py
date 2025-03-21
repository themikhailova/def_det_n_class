import cv2
import numpy as np
import time
import os
import sys
import pandas as pd


sys.path.append(os.path.abspath('./detect'))
from detect.features import calculate_features, save_features_to_excel
from detect.detect_anomalies import detect_and_highlight_anomalies, detect_differ
from detect.contours_connection import connect_contours, remove_nested_contours, merge_overlapping_contours
from detect.classif import classify_anomaly

sys.path.append(os.path.abspath('./preprocess'))
from preprocess.backremoveCV import remover
from preprocess.prep_metal import preprocess_metal_image
from preprocess.aligning import align

sys.path.append(os.path.abspath('./threed'))
from threed.rotate import find_temp

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
            all_cont_ing = input_image.copy()
            for contour in contours:
                
                cv2.drawContours(all_cont_ing, [contour], -1, (0, 0, 255), 2)
                if cv2.contourArea(contour) < min_anomaly_size:
                    continue
                
                # features_remove = calculate_features(contour, input_gray, max_area, max_perimeter, max_centroid)
                # # save_features_to_excel(features_remove, './spam/anomalies.xlsx')
                
                # if features_remove['relative_area'] < 0.0186:
                #     print(features_remove['relative_area'], features_remove['mean_intensity'])
                #     contours_new.append(contour)
            cv2.imwrite('./all_cont_ing.jpg', all_cont_ing)
            
            contours = merge_overlapping_contours(contours, input_gray, max_area, max_perimeter, max_centroid)
            contours = remove_nested_contours(contours, input_gray, max_area, max_perimeter, max_centroid)
        if mode==1:
            all_cont_ing = input_image.copy()
            contours_new = []
            for contour in contours:
                cv2.drawContours(all_cont_ing, [contour], -1, (0, 0, 255), 2)
                features_remove = calculate_features(contour, input_gray, max_area, max_perimeter, max_centroid)
                # save_features_to_excel(features_remove, './spam/anomalies.xlsx')
                
                if features_remove['relative_area'] < 0.01:
                    print(features_remove['relative_area'], features_remove['mean_intensity'])
                    contours_new.append(contour)
            contours = contours_new
            cv2.imwrite('./all_cont_ing.jpg', all_cont_ing)
        
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        test = input_image.copy()
        anomaly_index = 0
        if not contours:
            raise ValueError("Ошибка: контуры не найдены или пусты")
        
        columns = [
        "image_path", "part_number", "anomaly_type", "contour", "bounding_rect_x", "bounding_rect_y", "bounding_rect_w", "bounding_rect_h"
        ] + [
            "area", "perimeter", "relative_area", "relative_perimeter", "relative_centroid_distance", "compactness", "aspect_ratio", "eccentricity", "equivalent_diameter", "concavity", "angularity", "complexity"
        ] + [f"fourier_{i}" for i in range(2, 11)] + [
            "mean_intensity", "median_intensity", "std_intensity", "max_intensity", "min_intensity", "uniformity", "entropy", "mean_gradient", "std_gradient"
        ]

        # output_folder = "output_images"
        # os.makedirs(output_folder, exist_ok=True)

        excel_path = "./anomalies.xlsx"
        try:
            if os.path.exists(excel_path):
                df = pd.read_excel(excel_path)
            else:
                df = pd.DataFrame(columns=columns) 

        except PermissionError:
            print(f"Файл {excel_path} уже открыт. Закройте его и попробуйте снова.")
            df = None
        anomaly_index = 0  # Индекс аномалии

        for contour in contours:
            if cv2.contourArea(contour) < 15:
                continue
            output_image = input_image.copy()
            
            features = calculate_features(contour, input_gray, max_area, max_perimeter, max_centroid)
            
            anomaly_type, colour = classify_anomaly(features)
            if anomaly_type == 'no':
                colour = (0, 0, 255)
                # anomaly_type = 'Unknown'
            
            x, y, w, h = cv2.boundingRect(contour)
            
            cv2.drawContours(test, [contour], -1, colour, 2)
            # cv2.rectangle(output_image, (x, y), (x + w, y + h), colour, 2) 
            
            anomaly_filename = f"{output_folder}/{count_def}_anomaly_{anomaly_index}.png"
            cv2.imwrite(anomaly_filename, output_image)
            
            # Запись данных в DataFrame
            data = {
                "image_path": img_path,
                "anomaly_filename": anomaly_filename,
                "part_number": count_def,
                "Y": anomaly_type,
                "anomaly_colour": colour,
                "contour": str(contour.tolist()),
                "bounding_rect_x": x,
                "bounding_rect_y": y,
                "bounding_rect_w": w,
                "bounding_rect_h": h
            }
            
            # Добавляем характеристики
            for key in features.keys():
                data[key] = features[key]
            
            if df is None or df.empty or df.isna().all().all():
                df = pd.DataFrame([data])  # Если df пустой или содержит только NaN, создаем новый
            else:
                df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)
            
            anomaly_index += 1

        # Сохранение в Excel
        if os.path.exists(excel_path):
            with pd.ExcelWriter(excel_path, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
                df.to_excel(writer, index=False, sheet_name='Sheet1')
        else:
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Sheet1')

        print(f"Файл сохранен: {excel_path}")

        # for contour in contours:
        #     if cv2.contourArea(contour) < 15:
        #         continue
        #     output_image = input_image.copy()
            
        #     features = calculate_features(contour, input_gray, max_area, max_perimeter, max_centroid)

        #     anomaly_type, colour = None, None
        #     if anomaly_type is None:
        #         colour = (0, 0, 255)
        #         anomaly_type = 'Unknown' 
            
        #     x, y, w, h = cv2.boundingRect(contour)
          
        #     cv2.drawContours(test, [contour], -1, (0, 0, 255), 2)
        #     cv2.rectangle(output_image, (x, y), (x + w, y + h), colour, 2) 
        #     # cv2.rectangle(test, (x, y), (x + w, y + h), colour, 2) 
        #     anomaly_filename = f"{output_folder}/{count_def}_anomaly_{anomaly_index}.png"
        #     cv2.imwrite(anomaly_filename, align(output_image))
            
        #     anomaly_index += 1
         
        cv2.imwrite('./test.jpg', test)

def remove_back_and_detect(img_path, reference_path, output_folder, count_def, mode=0):
    
    input_image = remover(cv2.imread(img_path))
    reference_image = remover(cv2.imread(reference_path))
    cv2.imwrite('./input_img.png', input_image)
    start = time.time()
    detect_and_save_anomalies(img_path, input_image, reference_image, output_folder, mode, count_def)
    finish = time.time()

    res_msec = (finish - start) * 1000
    print('Время работы в миллисекундах: ', res_msec)

def set_reference_path(defect_path, template_directory):
    '''
    Функция принимает путь к директории из main.py и выполняет заглушку.
    :param defect_path: Path - путь, переданный из главного окна
    :param template_directory: Path - путь к папке с шаблонами
    '''
    for temp_file in os.listdir(template_directory):
        ref_path = os.path.join(template_directory, temp_file)

    if defect_path:
        print(f"Заглушка: Путь к директории успешно получен: {defect_path}")
        print(f"Директория шаблона: {template_directory}")
        
        output_folder = r'C:/output_folder'
        
        # Цикл по всем файлам в папке
        count_def = 0
        for file in os.listdir(defect_path):
            # Полный путь к файлу
            img_path = os.path.join(defect_path, file)
            # Проверяем, что это файл, а не папка
            if os.path.isfile(img_path):
                    count_def += 1
                    # Проверяем, что temp_file является файлом и имеет допустимое расширение
                    if os.path.isfile(ref_path) and ref_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                        remove_back_and_detect(img_path, ref_path, output_folder, count_def)
                    elif os.path.isfile(ref_path) and ref_path.lower().endswith(('.obj', '.stl')):
                        ref_img_path = find_temp(img_path, ref_path)
                        remove_back_and_detect(img_path, ref_img_path, output_folder, count_def, mode=1)
    else:
        print("Заглушка: Ошибка - путь к директории не передан")

# if __name__ == "__main__":
#     test_img_path = r"C:/det2/"
#     test_ref_path = r"C:/ref2/"
#     set_reference_path(test_img_path, test_ref_path)

# img_path = r'./def52.jpg'
# reference_path = r'./ref5.jpg'
# output_folder = r'./output_folder'

# remove_back_and_detect(img_path, reference_path, output_folder)
