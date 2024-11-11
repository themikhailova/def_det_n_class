import cv2
import os
import pickle

class DescriptorExtractor:
    def __init__(self, main_dir, desc_dir, detector_type):
        '''
        инициализация

        :param main_dir: путь к  каталогу с изображениями
        :param desc_dir: каталог для дескрипторов
        :param detector_type: тип детектора ('SIFT' или 'ORB')

        SIFT (Scale-Invariant Feature Transform): выделяет ключевые точки в изображении и описывает их характеристики, 
                                                  такие как угол ориентации, масштаб, текстурные особенности
        ORB (Oriented FAST and Rotated BRIEF): сочетает быстрый алгоритм FAST для обнаружения ключевых точек и BRIEF для их описания
                                                  ORB быстрее и подходит для работы в реальном времени
        '''
        self.main_dir = main_dir
        self.desc_dir = desc_dir
        self.detector = self._initialize_detector(detector_type)
        
        os.makedirs(desc_dir, exist_ok=True)

    def _initialize_detector(self, detector_type):
        '''
        инициализация алгоритма детектора

        :param detector_type: тип детектора ('SIFT' или 'ORB')
        :return: инициализированный детектор
        '''
        if detector_type == 'SIFT':
            return cv2.SIFT_create()
        elif detector_type == 'ORB':  
            return cv2.ORB_create()
        else:
            raise ValueError("Неподдерживаемый тип детектора. Используйте 'SIFT' или 'ORB'")

    def process_and_save_descriptors(self, image_path, save_path):
        '''
        обработка и сохранение дескрипторов

        :param image_path: путь к изображению
        :param save_path: путь для дескрипторов
        '''
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        keypoints, descriptors = self.detector.detectAndCompute(image, None)
        
        with open(save_path, 'wb') as f:
            pickle.dump(descriptors, f)

    def process_all_images(self):
        '''
        проход по всем подкаталогам и изображениям, обрабатка и сохранение дескрипторов
        '''
        for class_folder in os.listdir(self.main_dir):
            class_path = os.path.join(self.main_dir, class_folder)
            
            if not os.path.isdir(class_path):
                continue
            
            save_class_dir = os.path.join(self.desc_dir, class_folder)
            os.makedirs(save_class_dir, exist_ok=True)

            for image_file in os.listdir(class_path):
                image_path = os.path.join(class_path, image_file)
                
                if not image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    continue

                descriptor_save_path = os.path.join(save_class_dir, image_file + '.pkl')
                
                self.process_and_save_descriptors(image_path, descriptor_save_path)
                print(f"Descriptors for {image_file} saved.")

if __name__ == "__main__":
    main_dir = "./processed_details/"
    desc_dir = "./descriptors/"
    
    processor = DescriptorExtractor(main_dir, desc_dir, detector_type='SIFT')
    processor.process_all_images()
