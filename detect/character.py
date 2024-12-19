import cv2
import numpy as np
import time

from features import calculate_features, save_features_to_excel
from classify_anomalies import classify_anomaly
from detect_anomalies import detect_and_highlight_anomalies, detect_differ
from contours_connection import connect_contours, union_contours, remove_nested_contours, increase_ellipse


def detect_and_save_anomalies(input_image, reference_image, output_folder, threshold=50, region_size=0.05, min_anomaly_size=20, dilate_iter=5):
    """
    Основной алгоритм обнаружения аномалий и их сохранения в виде отдельных изображений.
    """  
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
        reference_gray2 = cv2.GaussianBlur(reference_gray, (5, 5), 0)
        input_gray2 = cv2.GaussianBlur(input_gray, (7, 7), 0)
        
        _, detailed_mask = detect_and_highlight_anomalies(input_gray)
        _, mask_ref = detect_and_highlight_anomalies(reference_gray, pixel_diff_threshold=20)
        mask_ref = cv2.bitwise_not(mask_ref)
        detailed_mask_combined = cv2.bitwise_and(detailed_mask, mask_ref)
        
        _, differ = detect_differ(input_gray2, reference_gray2, region_size_ratio=0.1)
        # _, differ = detect_differ(input_gray, reference_gray, region_size_ratio=0.1, pixel_diff_threshold=115)
        combined_anomalies = cv2.bitwise_or(detailed_mask_combined, differ)
        contours, _ = cv2.findContours(combined_anomalies, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
      
        contours_new = []
        for contour in contours:
            
            if cv2.contourArea(contour) < min_anomaly_size:
                continue
            if len(contour) < 5:
                continue
            features_remove = calculate_features(contour, input_gray, max_area, max_perimeter, max_centroid)
            # save_features_to_excel(features_remove, './spam/anomalies.xlsx')
            if features_remove['mean_intensity'] > 60:
                contours_new.append(contour)
        
        contours = contours_new   
        contours = connect_contours(contours, input_gray, max_area, max_perimeter, max_centroid, min_dist=15)
        contours = connect_contours(contours, input_gray, max_area, max_perimeter, max_centroid, min_dist=15)
        # # contours = connect_contours(contours, input_gray, max_area, max_perimeter, max_centroid)
        # # contours = connect_contours(contours, input_gray, max_area, max_perimeter, max_centroid)

        # # # contours, flag = union_contours(contours, input_gray, max_area, max_perimeter, max_centroid)

        # # # while flag:
        # # #     contours, flag = union_contours(contours, input_gray, max_area, max_perimeter, max_centroid)
        contours = remove_nested_contours(contours, input_gray, max_area, max_perimeter, max_centroid)
    
        test = input_image.copy()
        anomaly_index = 0
        for contour in contours:
            if cv2.contourArea(contour) < 20:
                continue
            if len(contour) < 5:
                continue
            # test = input_image.copy()
            output_image = input_image.copy()

            features = calculate_features(contour, input_gray, max_area, max_perimeter, max_centroid)

            anomaly_type, colour = classify_anomaly(features, contour, input_gray)
            if anomaly_type is None:
                colour = (255,255,255)
                anomaly_type = 'Unknown' 
            
            x, y, w, h = cv2.boundingRect(contour)
            ellipse = cv2.fitEllipse(contour)
            leftmost = tuple(contour[contour[:, :, 0].argmin()][0])
            rightmost = tuple(contour[contour[:, :, 0].argmax()][0])
            topleft = tuple(contour[contour[:, :, 1].argmin()][0])
            bottomright = tuple(contour[contour[:, :, 1].argmax()][0])
            
            center = ((leftmost[0] + rightmost[0]) // 2, (topleft[1] + bottomright[1]) // 2)
            axes_length = ((rightmost[0] - leftmost[0]) // 2), ((bottomright[1] - topleft[1]) // 2)
            
            
            if (np.all(np.isfinite(ellipse[0])) and
                np.all(np.isfinite(ellipse[1])) and
                ellipse[1][0] > 0 and ellipse[1][1] > 0):
                cv2.drawContours(test, [contour], -1, (0, 0, 255), 2)
                # cv2.putText(output_image, anomaly_type, (x, y - 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, colour, 2)
                # Обводим контур прямоугольником
                cv2.rectangle(output_image, (x, y), (x + w, y + h), colour, 2) # цвет и толщина
                # cv2.drawContours(output_image, [contour], -1, (0, 0, 255), 2)
                cv2.rectangle(test, (x, y), (x + w, y + h), colour, 2) # цвет и толщина
                # cv2.ellipse(output_image, center, axes_length, angle=0, startAngle=0, endAngle=360, color=colour, thickness=2)
                # cv2.ellipse(output_image, ellipse, colour, 2)
                anomaly_filename = f"{output_folder}/anomaly_{anomaly_index}.png"
                print(anomaly_filename)
                cv2.imwrite(anomaly_filename, output_image)
                # cv2.imshow('output_image', output_image)
                # cv2.imshow('test', test)
                # cv2.waitKey(0)
                anomaly_index += 1
            else:
                print(f"Некорректный эллипс: {ellipse}")
        cv2.imshow('test', test)
        cv2.waitKey(0)
    return output_image

input_image = cv2.imread("./5.jpg")
reference_image = cv2.imread("./6.jpg")
output_folder = "./output_anomalies/"

start = time.time()
output_image = detect_and_save_anomalies(input_image, reference_image, output_folder)
finish = time.time()

res_msec = (finish - start) * 1000
print('Время работы в миллисекундах: ', res_msec)