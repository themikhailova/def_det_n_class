import cv2
import numpy as np
# import matplotlib.pyplot as plt
import time

# image = cv2.imread(r"./def5.jpg")

def remover(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_image = cv2.GaussianBlur(hsv_image, (15,15), 0)

    lower_green = np.array([0, 100, 0])  # Нижний порог для зелёного
    upper_green = np.array([240, 255, 255])  # Верхний порог для зелёного

    hsv_image = cv2.GaussianBlur(hsv_image, (5, 5), 0)
    background_mask = cv2.inRange(hsv_image, lower_green, upper_green)
    foreground_mask = cv2.bitwise_not(background_mask)
    kernel = np.ones((15, 15), np.uint8)  # можно использовать, чтобы заполнить небольшие проплешины
    foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_CLOSE, kernel)

    result_image = cv2.bitwise_and(image, image, mask=foreground_mask)
    return result_image


# cv2.imwrite('./def55.jpg', result_image)
