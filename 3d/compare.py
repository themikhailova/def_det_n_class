import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import io

input_model_img = './front.jpg'
input_img = 'try11.jpg'
model_image = cv2.imread(input_model_img, cv2.IMREAD_GRAYSCALE)
target_image = cv2.imread(input_img, cv2.IMREAD_GRAYSCALE)

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

def mask_creation(img, threshold=30):
    '''
    Создание маски
    :param img: Исходное изображение
    :param threshold: порог для первой бинаризации
    :return: обрезанная маска
    '''
    _, target_binary = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, target_binary = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(target_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        # cv2.drawContours(img, [largest_contour], -1, (255), thickness=2)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        target_binary = cv2.morphologyEx(target_binary, cv2.MORPH_CLOSE, kernel)
        img = crop_to_contour(target_binary, largest_contour)
    return img

def resize_img(img, model, flag):
    '''
    Изменение размера изображение маски в соответсвии с размером исходного фото
    :param img: Исходное изображение
    :param model: изображение модели
    :param flag: флаг определения наибольшей стороны
    :return: масштабированное изображение
    '''
    # поиск контуров на изображении для центрирования маски
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # самый крупный контур
        largest_contour = max(contours, key=cv2.contourArea)
        _, _, contour_width, contour_height = cv2.boundingRect(largest_contour)
        # база для ресайзнутого изображения
        mask = np.zeros((contour_height, contour_width), dtype=np.uint8)

        # размеры изображения модели        
        img_h, img_w = model.shape
        # если ширина больше высоты, то ширину увеличиваем на 1, высоту на отношение высоты к ширине. если высота больше то наоборот (+ отношение ширины к высоте)
        if flag == 2:
            otn = img_h/img_w
            step_h = otn
            step_w = 1
        else:
            otn = img_w/img_h
            step_w = otn
            step_h = 1
        # увеличиваем размеры,  пока один из параметров не будет равен целевому
        wh = True
        while wh:
            img_w = img_w + step_w
            img_h = img_h + step_h
            if contour_height > int(img_h + step_h) and contour_width > int(img_w + step_w):
                wh = True
            else: wh = False

        img_h = int(img_h)
        img_w = int(img_w)
        model = cv2.resize(model, (img_w, img_h), interpolation=cv2.INTER_AREA)
        center_x, center_y = (contour_width // 2, contour_height // 2)
        mask_x, mask_y = (img_w // 2, img_h // 2)
        
        # вычисление смещения маски относительно центра изображения
        top_left_x = center_x - mask_x
        top_left_y = center_y - mask_y
        # наложение сглаженной маски
        mask[top_left_y:top_left_y + img_h, top_left_x:top_left_x + img_w] = model

    return mask

def difference(model_image, target_image):
    cropped_model_mask = mask_creation(model_image)
    cropped_img_mask = mask_creation(target_image)

    model_mask_h, model_mask_w = cropped_model_mask.shape
    img_mask_h, img_mask_w = cropped_img_mask.shape

    if((model_mask_h > model_mask_w and img_mask_h > img_mask_w) or (model_mask_h < model_mask_w and img_mask_h < img_mask_w)):
        if (model_mask_h > model_mask_w and img_mask_h > img_mask_w):
            flag = 1
        else: flag = 2
        resized_model = resize_img(cropped_img_mask, cropped_model_mask, flag)
        # cv2.imshow('resized_model', resized_model)
        # cv2.imshow('cropped_img_mask', cropped_img_mask)
        cv2.bitwise_not(cropped_img_mask)
        dif1 = cv2.bitwise_and(cropped_img_mask, cv2.bitwise_not(resized_model))
        dif2 = cv2.bitwise_and(resized_model, cv2.bitwise_not(cropped_img_mask))
        dif = cv2.bitwise_or(dif1, dif2)
        # cv2.imshow('dif', dif)
        # cv2.waitKey(0)
        return dif, np.sum(dif).item()
    else:
        print('100% not same')
        dif = None
        return dif, dif

# dif = difference(model_image, target_image)
# print(type(dif))