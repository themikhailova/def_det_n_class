import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

# на 1ом и 4ом траблы, на 6ом тоже не очень
image = cv2.imread("ourdets/blue/2.jpg")
start = time.time()

hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
hsv_image = cv2.GaussianBlur(hsv_image, (11,11), 0)

# Установка диапазона цвета для выделения фона (синий в данном случае)
lower_blue = np.array([100, 50, 50])  # Нижний порог (темно-синий)
upper_blue = np.array([140, 255, 255])  # Верхний порог (светло-синий)

background_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)
foreground_mask = cv2.bitwise_not(background_mask)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

filtered_gray = cv2.bitwise_and(gray_image, gray_image, mask=foreground_mask)
_, binary = cv2.threshold(filtered_gray, 90, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

mask_holes = np.zeros_like(gray_image)
mask_outer = np.zeros_like(gray_image)

for i, contour in enumerate(contours):
    if hierarchy[0][i][3] == -1:  # Внешний контур
        cv2.drawContours(mask_outer, [contour], -1, color=255, thickness=cv2.FILLED)
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

result_image = cv2.bitwise_and(image, image, mask=combined_mask)

finish = time.time()
res_msec = (finish - start) * 1000
print('Время работы в миллисекундах: ', res_msec)

plt.figure(figsize=(10, 10))
plt.imshow(foreground_mask, cmap='gray')
plt.title('Outer Contour')
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
