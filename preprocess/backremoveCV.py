import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from rembg import remove
from PIL import Image


def calculate_contrast_range(image):
    # Предполагается, что изображение уже в оттенках серого (0–255)
    contrast = np.std(image)
    return contrast


input_image = Image.open("42.jpg")


# output_image = remove(input_image)
# black_background = Image.new("RGB", output_image.size, (0, 0, 0))
# black_background.paste(output_image, (0, 0), output_image)

# black_background.save("42noback.png")


image = cv2.imread("1.jpg", cv2.IMREAD_GRAYSCALE)
start = time.time()
_, binary = cv2.threshold(image, 115, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

mask_holes = np.zeros_like(image)
mask_outer = np.zeros_like(image)

# Перебор всех контуров для поиска внешних и вложенных контуров
for i, contour in enumerate(contours):
    # Проверяем, что это внешний контур
    if hierarchy[0][i][3] == -1:
        # Рисуем внешний контур на отдельной маске
        cv2.drawContours(mask_outer, [contour], -1, color=255, thickness=cv2.FILLED)
        
        # Ищем вложенные контуры (отверстия)
        stack = [i]  # Начинаем с внешнего контура
        while stack:
            current_idx = stack.pop()
            # Ищем дочерние контуры для текущего
            for j, child_contour in enumerate(contours):
                if hierarchy[0][j][3] == current_idx:  # Родительский контур — текущий
                    if cv2.contourArea(child_contour) > 500:
                        # Если контур является вложенным, добавляем его в стек
                        stack.append(j)
                        # Рисуем этот вложенный контур на маске
                        cv2.drawContours(mask_holes, [child_contour], -1, color=255, thickness=cv2.FILLED)

# Инвертируем маску с отверстиями, чтобы отверстия стали черными
mask_holes = cv2.bitwise_not(mask_holes)
combined_mask = cv2.bitwise_and(mask_outer, mask_holes)
combined_image = cv2.bitwise_and(image, combined_mask)
finish = time.time()
res = finish - start
res_msec = res * 1000
print('Время работы в миллисекундах: ', res_msec)
# Показать результат для отверстий
plt.figure(figsize=(10, 10))
plt.imshow(mask_holes, cmap='gray')
plt.title('Holes')
plt.axis('off')
plt.show()

# Показать результат для внешнего контура
plt.figure(figsize=(10, 10))
plt.imshow(mask_outer, cmap='gray')
plt.title('Outer Contour')
plt.axis('off')
plt.show()

# Совмещение масок (внешний контур и отверстия)


# Показать результат для совмещенной маски
plt.figure(figsize=(10, 10))
plt.imshow(combined_mask, cmap='gray')
plt.title('Combined Mask: Outer Contour and Holes')
plt.axis('off')
plt.show()



# Показать результат для совмещенной маски
plt.figure(figsize=(10, 10))
plt.imshow(combined_image, cmap='gray')
plt.title('Combined Mask: Outer Contour and Holes')
plt.axis('off')
plt.show()
