import cv2
import numpy as np
import pandas as pd

def save_features_to_excel(features_list, filename):
    """
    Save a list of feature dictionaries to an Excel file.

    :param features_list: List of feature dictionaries.
    :param filename: Path to the output Excel file.
    """
    data = {
            'area': [features_list['area']],
            'perimeter': [features_list['perimeter']],
            'relative_area': [features_list['relative_area']],
            'relative_perimeter': [features_list['relative_perimeter']],
            'relative_centroid_distance': [features_list['relative_centroid_distance']],
            'compactness': [features_list['compactness']],
            'aspect_ratio': [features_list['aspect_ratio']],
            'eccentricity': [features_list['eccentricity']],
            'equivalent_diameter': [features_list['equivalent_diameter']],
            'concavity': [features_list['concavity']],
            'angularity': [features_list['angularity']],
            'complexity': [features_list['complexity']],
            'fourier_2': [features_list['fourier_2']],
            'fourier_3': [features_list['fourier_3']],
            'fourier_4': [features_list['fourier_4']],
            'fourier_5': [features_list['fourier_5']],
            'fourier_6': [features_list['fourier_6']],
            'fourier_7': [features_list['fourier_7']],
            'fourier_8': [features_list['fourier_8']],
            'fourier_9': [features_list['fourier_9']],
            'fourier_10': [features_list['fourier_10']],
            'mean_intensity': [features_list['mean_intensity']],
            'median_intensity': [features_list['median_intensity']],
            'std_intensity': [features_list['std_intensity']],
            'max_intensity': [features_list['max_intensity']],
            'min_intensity': [features_list['min_intensity']],
            'uniformity': [features_list['uniformity']],
            'entropy': [features_list['entropy']],
            'mean_gradient': [features_list['mean_gradient']],
            'std_gradient': [features_list['std_gradient']]
            }
    try:
                # Проверяем, существует ли файл
        existing_data = pd.read_excel(filename)
                # # df = pd.DataFrame({'Compactness': [features['compactness']],
                # #                     'Eccentricity': [features['eccentricity']],
                # #                     'Aspect Ratio': [features['aspect_ratio']],
                # #                     'Mean Intensity': [features['mean_intensity']],
                # #                     'Curvature': features['curvature'],
                # #                     'Bounding rect': features['bounding_rect'],
                # #                     'Min circle': features['min_circle']})
                # data = pad_dict_list(data, 0)

                # df = pd.DataFrame.from_dict(data, orient='index')
        df = pd.concat([existing_data, pd.DataFrame(data)], ignore_index=True)
    except FileNotFoundError:
        # Если файла нет, создаем новый
        df = pd.DataFrame(data)

            # Сохраняем в файл
    df.to_excel(filename, index=False)

def calculate_features(contour, image_gray, max_area, max_perimeter, max_centroid):
    features = {}

    # Area and perimeter
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    features['area'] = area
    features['perimeter'] = perimeter

    # Relative area
    features['relative_area'] = area / max_area if max_area > 0 else 0

    # Relative perimeter
    features['relative_perimeter'] = perimeter / max_perimeter if max_perimeter > 0 else 0

    # Centroid
    M = cv2.moments(contour)
    if M['m00'] != 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
    else:
        cx, cy = 0, 0
    centroid = (cx, cy)

    # Relative centroid distance
    if max_centroid is not None:
        dist_to_max_centroid = np.sqrt((centroid[0] - max_centroid[0])**2 + (centroid[1] - max_centroid[1])**2)
    else:
        dist_to_max_centroid = 0
    features['relative_centroid_distance'] = dist_to_max_centroid

    # Compactness
    if area > 0:
        compactness = (perimeter ** 2) / (4 * np.pi * area)
        features['compactness'] = compactness
    else:
        features['compactness'] = 0

    # Bounding rectangle aspect ratio
    x, y, w, h = cv2.boundingRect(contour)
    features['aspect_ratio'] = w / h if h > 0 else 0

    # Ellipse features (if enough points)
    if len(contour) >= 5:
        ellipse = cv2.fitEllipse(contour)
        (center, axes, angle) = ellipse
        major_axis = max(1, max(axes))
        minor_axis = min(axes)
        features['eccentricity'] = np.sqrt(1 - (minor_axis**2 / major_axis**2))
    else:
        features['eccentricity'] = 0

    # Equivalent diameter
    features['equivalent_diameter'] = np.sqrt(4 * area / np.pi) if area > 0 else 0

    # Convexity and concavity
    hull = cv2.convexHull(contour, returnPoints=False)
    if len(hull) > 0:
        try:
            defects = cv2.convexityDefects(contour, hull)
            convexity = cv2.isContourConvex(contour)
            features['convexity'] = int(convexity)

            if defects is not None:
                concavity = np.sum(defects[:, 0, 3]) / 256.0
                features['concavity'] = concavity
            else:
                features['concavity'] = 0
        except cv2.error:
            features['convexity'] = 0
            features['concavity'] = 0
    else:
        features['convexity'] = 0
        features['concavity'] = 0

    # Angularity
    epsilon = 0.02 * perimeter if perimeter > 0 else 1.0
    approx = cv2.approxPolyDP(contour, epsilon, True)
    features['angularity'] = len(approx)

    # Complexity
    features['complexity'] = len(contour) / perimeter if perimeter > 0 else 0

    # Fourier descriptors
    contour_np = contour[:, 0, :]
    contour_complex = np.empty(contour_np.shape[0], dtype=complex)
    contour_complex.real = contour_np[:, 0]
    contour_complex.imag = contour_np[:, 1]
    fourier_result = np.fft.fft(contour_complex)

    # Use the first few Fourier coefficients
    num_coefficients = 10
    for i in range(num_coefficients):
        features[f'fourier_{i+1}'] = 0.0  # Заполнение нулями
    fourier_magnitude = np.abs(fourier_result[:num_coefficients])
    for i, coeff in enumerate(fourier_magnitude):
        features[f'fourier_{i+1}'] = coeff

    # Mean intensity inside the contour
    mask = np.zeros_like(image_gray, dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
    mean_intensity = cv2.mean(image_gray, mask=mask)[0]
    features['mean_intensity'] = mean_intensity

    # Median intensity
    median_intensity = np.median(image_gray[mask == 255])
    features['median_intensity'] = median_intensity

    # Стандартное отклонение интенсивности (Standard Deviation of Intensity)
    std_intensity = np.std(image_gray[mask == 255])
    features['std_intensity'] = std_intensity

    # Max and min intensity
    max_intensity = np.max(image_gray[mask == 255])
    min_intensity = np.min(image_gray[mask == 255])
    features['max_intensity'] = max_intensity
    features['min_intensity'] = min_intensity
    # # Intensity histogram
    histogram = cv2.calcHist([image_gray], [0], mask, [256], [0, 256]).flatten()
    # features['intensity_histogram'] = histogram

    # Uniformity
    uniformity = np.sum((histogram / np.sum(histogram))**2)
    features['uniformity'] = uniformity

    # Entropy
    entropy = -np.sum((histogram / np.sum(histogram)) * np.log2(histogram / np.sum(histogram) + 1e-10))
    features['entropy'] = entropy

    # Gradient characteristics
    grad_x = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image_gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    mean_gradient = np.mean(gradient_magnitude[mask == 255])
    std_gradient = np.std(gradient_magnitude[mask == 255])
    features['mean_gradient'] = mean_gradient
    features['std_gradient'] = std_gradient
    # print(features)
    return features
