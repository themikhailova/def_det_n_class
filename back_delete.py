import cv2
import numpy as np
import os
from skimage.feature import local_binary_pattern

class ImageProcessor:
    def __init__(self, image_path):
        self.image_path = image_path
        

    def process_image(self):
        # обрабатывает одно изображение, сохраняет результат и возвращает промежуточные данные обработки'''
        image = cv2.imread(self.image_path)
        
        # преобразование в цветовое пространство LAB для улучшения выделения объектов и уменьшения влияния теней
        '''
        LAB — это цветовое пространство, используемое для представления цвета, которое состоит из трех компонентов:
        L (Lightness) — яркость или светлота, варьирующаяся от черного (0) до белого (100). Она описывает количество света в изображении
        A (Green-Red) — цветовой компонент, который меняется от зеленого к красному
        B (Blue-Yellow) — цветовой компонент, который изменяется от синего к желтому
        '''
        # выравнивание гистограммы (для уменьшения влияния теней)
        lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab_image) # разделение на 3 канала
        l = cv2.equalizeHist(l) # выравнивание гистограммы компонента L для улучшения яркости
        lab_image = cv2.merge((l, a, b)) # объединение обратно
        image_equalized = cv2.cvtColor(lab_image, cv2.COLOR_LAB2BGR) # преобразование обратно в цветовое пространство BGR

        # фильтр Гаусса для сглаживания изображения и уменьшения шума 
        '''
        Гауссов фильтр — это фильтр для размытия изображения, который используется для сглаживания и устранения шума
        Он применяет математическую функцию Гаусса для того, чтобы создать "размытое" изображение
        В зависимости от радиуса фильтра он может размыть изображение с разной степенью
        '''
        image_blurred = cv2.GaussianBlur(image_equalized, (5, 5), 0)

        # LBP для выделения текстурных признаков 
        '''
        Local Binary Pattern (LBP) — метод выделения текстурных признаков на изображении 
        Принцип работы LBP:
        Для каждого пикселя рассматриваются соседние пиксели в некотором радиусе
        Сравниваются значения интенсивности пикселя в центре и пикселей вокруг него
        Если значение интенсивности соседнего пикселя больше, чем у центрального, то этот пиксель кодируется единицей, иначе — нулем
        Полученное двоичное число интерпретируется как уникальный код, который описывает текстуру в этом местоположении
        LBP особенно полезен в выделении текстурных элементов, таких как края, линии и пятна
        '''
        lbp_image = self.apply_lbp(image_blurred)

        # сегментация с GrabCut и Watershed
        '''
        GrabCut — это метод сегментации изображений, который отделяет объекты на переднем плане от фона
        работает в два этапа:
        Инициализация прямоугольником: Устанавливается начальный прямоугольник, в котором должен находиться объект (у нас - почти все изображение)
        Алгоритм итеративного уточнения: Исходя из статистики цветовых моделей, он анализирует пиксели внутри и снаружи прямоугольника, 
        выделяя те, которые принадлежат фону, и те, которые могут быть частью объекта
        
        
        (из Яндекса) Алгоритм Watershed —  метод сегментации изображения, основанный на концепции водораздельного преобразования
        Принцип работы: изображение рассматривается как топографическая поверхность, на основе интенсивности пикселей определяются бассейны водосбора
        Локальные минимумы отмечаются как начальные точки, и бассейны заполняются цветами до достижения границ объектов. 
        Процесс алгоритма Watershed включает несколько шагов: 
        Размещение маркеров. На локальных минимумах изображения размещаются маркеры, которые служат начальными точками для процесса заполнения
        Заполнение. Затем алгоритм заливает изображение разными цветами, начиная от маркеров. По мере распространения цвет заполняет бассейны водосбора до достижения границ объектов или областей на изображении. 2
        Формирование бассейнов водосбора. По мере распространения цвета постепенно заполняются бассейны водосбора, создавая сегментацию изображения. Полученным сегментам или областям присваиваются уникальные цвета, которые затем можно использовать для идентификации разных объектов или особенностей на изображении. 2
        Идентификация границ. Алгоритм использует границы между разными цветными областями для идентификации объектов или областей на изображении.


        У нас этот алгоритм хорошо работает с тенями, поэтому его маску используем для выделения теней
        '''
        final_mask = self.apply_grabcut_and_watershed(image_blurred, image)

        # наложение маски и морфологическое сглаживание
        result = self.apply_mask_and_morphology(image, final_mask)
        
        # output_image_path = os.path.join(self.output_folder, output_filename)
        # cv2.imwrite(output_image_path, result)
        
        # print(f"Обработано изображение: {output_filename}")

        # результаты обработки
        return {
            "image_equalized": image_equalized, # изображение после выравнивания гистограммы
            "lbp_image": lbp_image, # изображение после применения lbp
            "final_mask": final_mask, # итоговая маска
            "result": result # итоговое изображение
        }

    def apply_lbp(self, image):
        ''' LBP для выделения текстурных признаков'''
        radius = 1
        n_points = 8 * radius
        lbp_image = local_binary_pattern(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), n_points, radius, method='uniform')
        return np.uint8(lbp_image)

    def apply_grabcut_and_watershed(self, image_blurred, image):
        '''методы GrabCut и Watershed для сегментации изображения'''
        mask = np.zeros(image_blurred.shape[:2], np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        rect = (10, 10, image_blurred.shape[1] - 10, image_blurred.shape[0] - 10)
        cv2.grabCut(image_blurred, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

        gray = cv2.cvtColor(image_blurred, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        kernel = np.ones((5, 5), np.uint8)
        sure_bg = cv2.dilate(thresh, kernel, iterations=5)
        
        dist_transform = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
        
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0

        markers = cv2.watershed(image, markers)
        watershed_mask = np.uint8(markers <= 1) * 255
        return mask2 & (watershed_mask // 255)

    def apply_mask_and_morphology(self, image, final_mask):
        '''наложение маски на изображение и сглаживание контуров с помощью морфологических операций'''
        result = image * final_mask[:, :, np.newaxis]
        # kernel = np.ones((5, 5), np.uint8)
        # result_smooth = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
        # result_smooth = cv2.morphologyEx(result_smooth, cv2.MORPH_OPEN, kernel)
        # return cv2.GaussianBlur(result_smooth, (5, 5), 0)
        return result

# output_folder = r'.\processed_details1'
# processor = ImageProcessor(output_folder)

# image_path = r'D:\Desktop\ref_training\details\2.jpg'
# output_filename = 'processed_example.jpg'
# results = processor.process_image(image_path, output_filename)

# image_equalized = results["image_equalized"]
# lbp_image = results["lbp_image"]
# final_mask = results["final_mask"]
# result = results["result"]

# cv2.imshow("res", result)
# cv2.waitKey(0)

