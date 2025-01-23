import cv2

def preprocess_metal_image(image, blur_ksize=(15, 15), alpha=0.55, beta=0.4):
    '''
    Предобработка изображения металлической детали для улучшения видимости дефектов

    Параметры:
    - image: входное изображение в формате BGR
    - blur_ksize: tuple, размер ядра для Gaussian Blur (low-pass фильтр)
    - alpha: вес для equalized изображения при комбинировании
    - beta: вес для high-pass фильтра при комбинировании

    Возвращает:
    - combined: финальное обработанное изображение
    '''
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # увеличение контрастности с помощью выравнивания гистограммы
    equalized = cv2.equalizeHist(gray)

    # 5. Подчёркивание локальных изменений (High-Pass Filter)
    low_pass = cv2.GaussianBlur(gray, blur_ksize, 0)
    high_pass = cv2.subtract(gray, low_pass)

    # 6. Создание комбинированного изображения
    combined = cv2.addWeighted(equalized, alpha, high_pass, beta, 0)

    return combined