import cv2
import numpy as np

from skimage.metrics import structural_similarity as ssim


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
            # print(np.mean(region_diff))
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

def detect_differ(obj_img, ref_img, region_size_ratio=0.05, anomaly_threshold=0.5, pixel_diff_threshold=40):

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