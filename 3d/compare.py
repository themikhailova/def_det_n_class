import numpy as np
import cv2
def crop_to_contour(image, contour):
    '''
    Обрезает изображение по ограничивающему прямоугольнику заданного контура
    :param image: Исходное изображение
    :param contour: Контур объекта
    :return: Обрезанное изображение
    '''
    x, y, contour_width, contour_height = cv2.boundingRect(contour)
    cropped_image = image[y:y+contour_height, x:x+contour_width]
    return cropped_image
def mask_creation(img, threshold=0):
    _, binary = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        return crop_to_contour(binary, largest_contour)
    else:
        raise ValueError("Contours not found in the image.")

def resize_img(target, model):
    target_h, target_w = target.shape
    model_h, model_w = model.shape
    scale = min(target_h / model_h, target_w / model_w)
    resized_model = cv2.resize(model, (int(model_w * scale), int(model_h * scale)), interpolation=cv2.INTER_AREA)
    mask = np.zeros_like(target, dtype=np.uint8)
    y_offset = (target_h - resized_model.shape[0]) // 2
    x_offset = (target_w - resized_model.shape[1]) // 2
    mask[y_offset:y_offset + resized_model.shape[0], x_offset:x_offset + resized_model.shape[1]] = resized_model
    return mask

def compute_difference(model, target):
    diff = cv2.absdiff(model, target)
    return np.sum(diff), diff

def difference(model_image, target_image):
    try:
        model_mask = mask_creation(model_image)
        target_mask = mask_creation(target_image)
        target_h, target_w = target_mask.shape
        model_h, model_w = model_mask.shape
        
        if(target_h >= target_w and model_h >= model_w) or (target_h <= target_w and model_h <= model_w):
            # cv2.imshow('model_mask', model_mask)
            # # cv2.imshow('target_mask', target_mask)
            # cv2.waitKey(0)
            print(target_mask.shape, model_mask.shape)
            resized_model = resize_img(target_mask, model_mask)
            total_diff, diff_matrix = compute_difference(resized_model, target_mask)
            return total_diff, diff_matrix
        else: return None, None
    except ValueError as e:
        print(f"Error: {e}")
        return None, None

# Использование
input_model_img = './front.jpg'
input_img = 'try11.jpg'

model_image = cv2.imread(input_model_img, cv2.IMREAD_GRAYSCALE)
target_image = cv2.imread(input_img, cv2.IMREAD_GRAYSCALE)

diff, diff_matrix = difference(model_image, target_image)
if diff is not None:
    print(f"Total difference: {diff}")
    cv2.imshow('Difference Matrix', diff_matrix)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
