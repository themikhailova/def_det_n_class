import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

# на 1ом и 4ом траблы, на 6ом тоже не очень
image = cv2.imread("ourdets/blue/2.jpg")
start = time.time()

hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
hsv_image = cv2.GaussianBlur(hsv_image, (11,11), 0)

# пороги синего работают нормально везде, кроме 6го фото (глубокая деталь с тенью внутрь)
lower_blue = np.array([100, 120, 0])  # Нижний порог (темно-синий) # изменение s с 50 на 120 оставит ненасыщ. блики на детали, а v с 50 на 0 уберет тени
upper_blue = np.array([140, 255, 255])  # Верхний порог (светло-синий)

# зеленый - менее универсальный. при следующих порогах на некоторых фото (пр. 4) не до конца убирает внутренние отверстия
# lower_green = np.array([0, 100, 0])  # Нижний порог для зелёного
# upper_green = np.array([120, 255, 255])  # Верхний порог для зелёного

# красный - худший вар. на глубокой детали убирает больше всего, на выпуклых (1 и 5) берет боковую тень
# lower_red1 = np.array([0, 70, 70])  # Нижний порог для первого диапазона
# upper_red1 = np.array([50, 255, 255])  # Верхний порог для первого диапазона

# lower_red2 = np.array([50, 50, 0])  # Нижний порог для второго диапазона
# upper_red2 = np.array([180, 255, 255])  # Верхний порог для второго диапазона

# hsv_image = cv2.GaussianBlur(hsv_image, (5, 5), 0)
# # Создание масок для двух диапазонов красного
# mask_red1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
# mask_red2 = cv2.inRange(hsv_image, lower_red2, upper_red2)

# # Объединение масок
# background_mask = cv2.bitwise_or(mask_red1, mask_red2)
# foreground_mask = cv2.bitwise_not(background_mask)

background_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)
foreground_mask = cv2.bitwise_not(background_mask)
kernel = np.ones((15, 15), np.uint8)  # можно использовать, чтобы заполнить небольшие проплешины
foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_CLOSE, kernel)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

filtered_gray = cv2.bitwise_and(gray_image, gray_image, mask=foreground_mask)
_, binary = cv2.threshold(filtered_gray, 90, 255, cv2.THRESH_BINARY) # 90 заменить на 0, т.к. все лишнее было отфильтровано ранее
contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

mask_holes = np.zeros_like(gray_image)
mask_outer = np.zeros_like(gray_image)

for i, contour in enumerate(contours):
    if hierarchy[0][i][3] == -1:  # Внешний контур
        cv2.drawContours(mask_outer, [contour], -1, color=255, thickness=cv2.FILLED) # тогда здесь получится более цельный внешний контур
        # stack = [i]
        # while stack:
        #     current_idx = stack.pop()
        #     for j, child_contour in enumerate(contours):
        #         if hierarchy[0][j][3] == current_idx:
        #             if cv2.contourArea(child_contour) > 500:
        #                 stack.append(j)
        #                 cv2.drawContours(mask_holes, [child_contour], -1, color=255, thickness=cv2.FILLED)

mask_holes = cv2.bitwise_not(mask_holes)
combined_mask = cv2.bitwise_and(mask_outer, foreground_mask)

result_image = cv2.bitwise_and(image, image, mask=foreground_mask)

finish = time.time()
res_msec = (finish - start) * 1000
print('Время работы в миллисекундах: ', res_msec)

# Отображение результатов
plt.figure(figsize=(10, 10))
plt.imshow(foreground_mask, cmap='gray')
plt.title('Foreground Mask')
plt.axis('off')
plt.show()

plt.figure(figsize=(10, 10))
plt.imshow(mask_outer, cmap='gray')
plt.title('Outer Contour Mask')
plt.axis('off')
plt.show()

plt.figure(figsize=(10, 10))
plt.imshow(combined_mask, cmap='gray')
plt.title('Combined Mask')
plt.axis('off')
plt.show()

plt.figure(figsize=(10, 10))
plt.imshow(result_image, cmap='gray')
plt.title('result ')
plt.axis('off')
plt.show()
