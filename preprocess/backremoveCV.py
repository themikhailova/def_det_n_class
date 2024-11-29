import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

image = cv2.imread(r"./5.jpg")
start = time.time()

hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
hsv_image = cv2.GaussianBlur(hsv_image, (15,15), 0)

# # # Установка диапазона цвета для выделения фона (красный в данном случае)
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


lower_green = np.array([0, 140, 0])  # Нижний порог для зелёного
upper_green = np.array([220, 255, 255])  # Верхний порог для зелёного

# Установка диапазона цвета для выделения фона (синий в данном случае)
# lower_blue = np.array([100, 50, 0])  # Нижний порог (темно-синий) # изменение s с 50 на 120 оставит ненасыщ. блики на детали, а v с 50 на 0 уберет тени
# upper_blue = np.array([140, 255, 255])  # Верхний порог (светло-синий)
hsv_image = cv2.GaussianBlur(hsv_image, (5, 5), 0)
background_mask = cv2.inRange(hsv_image, lower_green, upper_green)
foreground_mask = cv2.bitwise_not(background_mask)
kernel = np.ones((15, 15), np.uint8)  # можно использовать, чтобы заполнить небольшие проплешины
foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_CLOSE, kernel)

result_image = cv2.bitwise_and(image, image, mask=foreground_mask)

finish = time.time()
res_msec = (finish - start) * 1000
print('Время работы в миллисекундах: ', res_msec)

plt.figure(figsize=(10, 10))
plt.imshow(foreground_mask, cmap='gray')
plt.title('Foreground Mask')
plt.axis('off')
plt.show()


plt.figure(figsize=(10, 10))
plt.imshow(result_image, cmap='gray')
plt.title('result ')
plt.axis('off')
plt.show()

cv2.imwrite('./refnoBack.jpg', result_image)
