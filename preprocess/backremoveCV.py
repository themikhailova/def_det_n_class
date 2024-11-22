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
start = time.time()

# output_image = remove(input_image)
# black_background = Image.new("RGB", output_image.size, (0, 0, 0))
# black_background.paste(output_image, (0, 0), output_image)

# black_background.save("42noback.png")


image = cv2.imread("1.jpg", cv2.IMREAD_GRAYSCALE)
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
combined_mask = cv2.bitwise_and(mask_outer, mask_holes)

# Показать результат для совмещенной маски
plt.figure(figsize=(10, 10))
plt.imshow(combined_mask, cmap='gray')
plt.title('Combined Mask: Outer Contour and Holes')
plt.axis('off')
plt.show()

combined_image = cv2.bitwise_and(image, combined_mask)

# Показать результат для совмещенной маски
plt.figure(figsize=(10, 10))
plt.imshow(combined_image, cmap='gray')
plt.title('Combined Mask: Outer Contour and Holes')
plt.axis('off')
plt.show()
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# import numpy as np
# import matplotlib.pyplot as plt
# from skimage import io, filters, morphology, exposure
# from skimage.color import rgb2gray
# from skimage.segmentation import chan_vese
# from skimage.filters import unsharp_mask
# import cv2
# from skimage.feature import local_binary_pattern


# # Функция для отображения изображений
# def show_images(images, titles, cmap=None):
#     plt.figure(figsize=(15, 5))
#     for i, img in enumerate(images):
#         plt.subplot(1, len(images), i + 1)
#         plt.imshow(img, cmap=cmap)
#         plt.title(titles[i])
#         plt.axis('off')
#     plt.tight_layout()
#     plt.show()

# # 1. Метод HSV (яркость)
# def segment_hsv(image):
#     hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#     v_channel = hsv[:, :, 2]  # Канал яркости
#     _, binary_hsv = cv2.threshold(v_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     return binary_hsv

# # 2. Пороговая обработка (Thresholding)
# def segment_threshold(gray_image):
#     _, binary_threshold = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     return binary_threshold

# def apply_lbp(self, image):
#         ''' LBP для выделения текстурных признаков'''
#         radius = 1
#         n_points = 8 * radius
#         lbp_image = local_binary_pattern(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), n_points, radius, method='uniform')
#         return np.uint8(lbp_image)
# # Главная функция
# def main():
#     image_path = "1111noback.jpg"  # Путь к изображению
#     image = cv2.imread(image_path)

#     # выравнивание гистограммы (для уменьшения влияния теней)
#     lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
#     l, a, b = cv2.split(lab_image) # разделение на 3 канала
#     l = cv2.equalizeHist(l) # выравнивание гистограммы компонента L для улучшения яркости
#     lab_image = cv2.merge((l, a, b)) # объединение обратно
#     image_equalized = cv2.cvtColor(lab_image, cv2.COLOR_LAB2BGR) # преобразование обратно в цветовое пространство BGR

    
#     image_blurred = cv2.GaussianBlur(image_equalized, (5, 5), 0)

#         # LBP для выделения текстурных признаков 
#     lbp_image = apply_lbp(image_blurred)
#     # Преобразование в градации серого
#     binary_hsv = segment_hsv(lbp_image)
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     # 1. Улучшение контраста
#     gray_image = exposure.adjust_gamma(gray, gamma=2.2)
#     gray_image = unsharp_mask(gray_image, radius=1.5, amount=1.5)

#     # 2. Удаление шума
#     blurred_image = filters.gaussian(gray_image, sigma=1.2)

#     # 3. Пороговая обработка
#     binary_image = blurred_image > filters.threshold_otsu(blurred_image)

#     # 4. Морфологическая очистка
#     binary_image = morphology.remove_small_objects(binary_image, min_size=200)
#     binary_image = morphology.remove_small_holes(binary_image, area_threshold=300)

#     # 5. Выделение контуров
#     edges = cv2.Canny((binary_image * 255).astype(np.uint8), threshold1=200, threshold2=150)



#     binary_threshold = segment_threshold(gray)
#     # Поиск контуров с иерархией
#     contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#     # Копия изображения для заливки
#     filled = image.copy()
#     largest_contour = max(contours, key=cv2.contourArea)
#     # Проходимся по всем контурам
#     for i, contour in enumerate(contours):
#         print(i)
#         # Проверяем, является ли текущий контур дочерним (имеет родителя)
#         if hierarchy[0][i][3] != 0:  # hierarchy[0][i][3] указывает на индекс родительского контура
#             # Проверка площади контура, чтобы игнорировать шум
#             if cv2.contourArea(contour) > 500:  # Порог площади
#                 cv2.drawContours(filled, [contour], -1, (0, 0, 0), thickness=cv2.FILLED)

#     # cv2.imshow("contours", filled)
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
#     binary_hsv = segment_hsv(image)
#     binary_threshold = segment_threshold(binary_hsv)

#     # Отображение результатов
#     show_images(
#         [image, edges, filled],
#         [
#             "Оригинал",
#             "canny",
#             "contours"
#         ],
#         cmap="gray",
#     )
#     cv2.imwrite('otu.jpg', binary_threshold)


# # Запуск программы
# if __name__ == "__main__":
#     main()

