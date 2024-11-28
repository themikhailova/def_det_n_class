import cv2
import numpy as np
import os

class ImageAugmentor:
    def __init__(self, augmented_path='.\\augmentation\\augmented_details'):
        '''
        Инициализация класса аугментации.

        :param augmented_path: путь к каталогу для сохранения аугментированных изображений.
        '''
        self.augmented_path = augmented_path
        os.makedirs(augmented_path, exist_ok=True)

    def augment_image(self, image, filename):
        '''
        Применение различных аугментаций к изображению.

        :param image: исходное изображение.
        :param filename: имя файла для сохранения аугментированного изображения.
        :return: список аугментированных изображений.
        '''
        augmented_images = []
        
        # 1. Изменение яркости и контраста
        bright_contrast_img = self.adjust_brightness_contrast(image)
        augmented_images.append(bright_contrast_img)

        # 2. Масштабирование изображения
        scaled_img = self.scale_image(image)
        augmented_images.append(scaled_img)

        # 3. Добавление шума
        noisy_img = self.add_noise(image)
        augmented_images.append(noisy_img)

        # 4. Поворот изображения
        rotated_img = self.rotate_image(image)
        augmented_images.append(rotated_img)

        # Сохранение аугментированных изображений
        self.save_images(augmented_images, filename)
        return augmented_images

    def adjust_brightness_contrast(self, image):
        '''
        Регулировка яркости и контраста изображения.

        :param image: исходное изображение.
        :return: изображение с изменёнными яркостью и контрастом.
        '''
        alpha = np.random.uniform(0.8, 1.2)  # Коэффициент контраста.
        beta = np.random.randint(-10, 10)    # Сдвиг яркости.
        return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    def scale_image(self, image):
        '''
        Масштабирование изображения с сохоанением исходного размера.

        :param image: исходное изображение.
        :return: масштабированное изображение.
        '''
        scale = np.random.uniform(0.9, 1.1)
        h, w = image.shape[:2]
        scaled_img = cv2.resize(image, (int(w * scale), int(h * scale)))
        return cv2.resize(scaled_img, (w, h))  # Возвращаем к оригинальному размеру.

    def add_noise(self, image):
        '''
        Добавление случайного шума к изображению.

        :param image: исходное изображение.
        :return: изображение с добавленным шумом.
        '''
        noise = np.random.normal(0, 10, image.shape).astype(np.uint8)
        return cv2.add(image, noise)

    def rotate_image(self, image):
        '''
        Поворот изображения на случайный угол.

        :param image: исходное изображение.
        :return: повернутое изображение.
        '''
        angle = np.random.uniform(-10, 10)
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
        return cv2.warpAffine(image, M, (w, h))

    def save_images(self, images, filename):
        '''
        Сохранение списка изображений в подкаталог по указанному пути.

        :param images: список аугментированных изображений.
        :param filename: базовое имя файла.
        '''
        for count, img in enumerate(images, start=1):
            aug_filename = f"{filename}_aug{count}.jpg"
            aug_filepath = os.path.join(self.augmented_path, aug_filename)
            cv2.imwrite(aug_filepath, img)
            print(f"Saved: {aug_filepath}")


# if __name__ == "__main__":
#     label = '1'
#     filename = 'image_example'
    
#     image = cv2.imread('./path/to/your/image.jpg')  # Путь к изображению
#     augmentor = ImageAugmentor()
#     augmented_images = augmentor.augment_image(image, label, filename)
