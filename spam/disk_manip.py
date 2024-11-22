import sys, re
sys.path.append('/'.join(re.split('[\\/]', __file__)[:-3]))

import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
# from src.config import Dir, Dataset, os_join
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LinearRegression
import math

from functools import reduce

from collections import deque

def make_dataset_from_img(img: cv2.Mat, xyFlag: bool=False) -> np.ndarray:
    '''
    Переводит изображение в формат датасета (колонки B, G, R)
    '''
    indices = np.indices(img.shape[:2]) # shape[0] строк в каждом []
    #           нормальный порядок: (0, 1, 2)
    colors = np.transpose(img, axes=(2, 0, 1)) # смена размерности так, чтобы было 3 строки BGR
    xycolors = np.stack((*indices, *colors), axis=-1)
    
    if xyFlag:
        datasetImg = np.reshape(xycolors, (-1, 5))
    else:
        datasetImg = np.reshape(img, (-1, 3))
    return datasetImg
        
def extract_disk_label(labels: np.ndarray, shape: (int)) -> int:
    '''
    Извлекает лэйбл диска из списка лэйблов `labels` для каринки 
    развмерностью `shape`
    
    Анализирует серединную 1/25 кадра, в предположении, 
    что большинство пикселей в этой области принадлежат диску
    '''
    # labels = np.reshape(labels, shape)
    xsize = shape[1]//5
    xs, xf = 2*xsize, 3*xsize
    ysize = shape[0]//5
    ys, yf = 3*ysize, 4*ysize
    
    print(np.unique(labels))
    print(np.max(labels[ys:yf, xs:xf]))
    
    return np.max(labels[ys:yf, xs:xf])

def extract_disk_from_labels(img: cv2.Mat, labels: np.ndarray) -> cv2.Mat:
    '''
    Извлекает диск из картинки на основе лэйблов кластеризации
    '''
    # labels = np.reshape(labels, img.shape[:2])
    diskLabel = extract_disk_label(labels, img.shape[:2])
    diskLocs = np.stack(np.where(labels==diskLabel), axis=-1)
    
    diskImg = np.zeros(img.shape)
    diskPixels = img[diskLocs[:, 0], diskLocs[:, 1]]
    diskImg[diskLocs[:, 0], diskLocs[:, 1]] = diskPixels
    
    diskImg = np.uint8(diskImg)
    return diskImg

def db_clustering(datasetImg: np.ndarray, img: cv2.Mat) -> (np.ndarray, cv2.Mat):
    ''' 
    Кластеризация методом DBSCAN 
    '''
    
    # print(int(0.003 * 0.4 * np.size(img)/3))
    db = DBSCAN(eps=5, min_samples=int(0.003 * 0.4 * np.size(img)/3), metric = 'euclidean',algorithm ='auto')
    db.fit(datasetImg)
    labels = db.labels_
    labels = np.reshape(labels, img.shape[:2])
    diskImg = extract_disk_from_labels(img, labels)
    return (labels, diskImg)

def em_clustering(datasetImg: np.ndarray, img: cv2.Mat) -> (np.ndarray, cv2.Mat):
    ''' 
    Кластеризация методом Expectation-maximization 
    '''
    em = GaussianMixture(n_components=2, init_params='kmeans')
    em.fit(datasetImg)
    labels = em.predict(datasetImg) 
    labels = np.reshape(labels, img.shape[:2])   
    diskImg = extract_disk_from_labels(img, labels)
    return (labels, diskImg)

def plot_results(img: cv2.Mat, labelsList: np.ndarray, diskImgs: cv2.Mat, methods: [str]):
    fig = plt.figure(layout='constrained')
    gs = GridSpec(2, 3)
    
    ax0 = fig.add_subplot(gs[:, 0])
    ax01 = fig.add_subplot(gs[0, 1])
    ax02 = fig.add_subplot(gs[0, 2])
    ax11 = fig.add_subplot(gs[1, 1])
    ax12 = fig.add_subplot(gs[1, 2])
    
    ax0.imshow(img[:, :, ::-1])
    ax0.set_title('Original image')
    
    ax01.imshow(labelsList[0])
    ax11.imshow(diskImgs[0][:, :, ::-1])
    ax01.set_title(methods[0])
    
    ax02.imshow(labelsList[1])
    ax12.imshow(diskImgs[1][:, :, ::-1])
    ax02.set_title(methods[1])
    
    for ax in [ax0, ax01, ax02, ax11, ax12]:
        ax.axis('off')
    
    plt.show()
    
        
def extract_disk(img: cv2.Mat) -> cv2.Mat:
    # приводим изображение к формату датасета
    datasetImg =  make_dataset_from_img(img, xyFlag=False)
    
    # кластеризация
    dbLabels, dbDiskImg = db_clustering(datasetImg, img)
    emLabels, emDiskImg = em_clustering(datasetImg, img)
    
    plot_results(img, [dbLabels, emLabels], [dbDiskImg, emDiskImg], ['DBSCAN', 'EM'])
    
    return dbDiskImg

def extract_disk_by_contours(img: cv2.Mat):
    gray = img
    if len(gray.shape) > 2:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def get_diagonal(minRect) :
    # if np.linalg.norm(p1-p2) > 368: print(np.linalg.norm(p1-p2))
    return np.linalg.norm(np.array(minRect[1]))


def extract_disk_by_inner_fill(img: cv2.Mat):
    imgHsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    canny =  cv2.Canny(imgHsv[:, :, 1], 200, 220)
    print(5)
    img = cv2.GaussianBlur(canny, (7, 7), 0)
    print(6)
    # img = cv2.medianBlur(canny, 3)
    img = np.array(img, np.int16)
    
    h, w = img.shape[:2] 
    h -= 1
    w -= 1
    
    threshold = 60

    def flood_fill(img, row, col, replacement=0):
        target = img[row, col].copy()
        
        def is_safe(r, c):
            return (0 <= r <= h and 0 <= c <= w) and np.mean(abs(target - img[r, c])) <= threshold
        
        if not is_safe(row, col):
            return img
        
        steps = [[0, 1], [1, 0], [0, -1], [-1, 0]]
        
        q = deque()
        q.append((row, col))
        visited = [(row, col)]
        
        while q:
            print(7)
            row, col = q.popleft()    
            img[row, col] = replacement
            
            for r, c in steps:
                if (row+r, col+c) not in visited and is_safe(row+r, col+c):
                    visited.append((row+r, col+c))
                    q.append((row+r, col+c))
            print(8)
        
        mask = np.ones(img.shape[:2], dtype=np.uint8) * 255
        visited = np.array(visited)
        mask[visited[:, 0], visited[:, 1]] = np.zeros(len(visited))
        
        return img, mask
    
    img, mask = flood_fill(img, h//2, w//2, replacement=255)
    
    mask = np.bitwise_xor(mask, np.ones_like(mask)*255)
    
    # imgMasks = list(zip(*imgMasks))
    # mask = reduce(lambda x, y: np.bitwise_and(x, y), imgMasks[1])


    img = np.array(img, np.uint8)
    
    return mask # cv2.cvtColor(imgHsv, cv2.COLOR_HSV2BGR)
    

def extract_disk_by_outer_fill(img: cv2.Mat):
    # imgHsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # np.array(img, np.int16)
    img = np.array(img, np.int16)
    
    h, w = img.shape[:2] 
    h -= 1
    w -= 1
    
    threshold = 20

    def flood_fill(img, row, col, replacement=0):
        target = img[row, col].copy()
        
        def is_safe(r, c):
            return (0 <= r <= h and 0 <= c <= w) and np.mean(abs(target - img[r, c])) <= threshold
        
        if not is_safe(row, col):
            return img
        
        steps = [[0, 1], [1, 0], [0, -1], [-1, 0]]
        
        q = deque()
        q.append((row, col))
        visited = [(row, col)]
        
        while q:
            row, col = q.popleft()    
            img[row, col] = replacement
            
            for r, c in steps:
                if (row+r, col+c) not in visited and is_safe(row+r, col+c):
                    visited.append((row+r, col+c))
                    q.append((row+r, col+c))
        
        mask = np.ones(img.shape[:2], dtype=np.uint8) * 255
        visited = np.array(visited)
        mask[visited[:, 0], visited[:, 1]] = np.zeros(len(visited))
        
        return img, mask
    
    img, mask00 = flood_fill(img, 0, 0)
    img, maskH0 = flood_fill(img, h, 0)
    img, mask0W = flood_fill(img, 0, w)
    img, maskHW = flood_fill(img, h, w)
    
    # imgMasks = list(zip(*imgMasks))
    # mask = reduce(lambda x, y: np.bitwise_and(x, y), imgMasks[1])
    
    mask = np.bitwise_and(mask00, maskH0)
    mask = np.bitwise_and(mask, mask0W)
    mask = np.bitwise_and(mask, maskHW)

    img = np.array(img, np.uint8)
    
    return mask # cv2.cvtColor(imgHsv, cv2.COLOR_HSV2BGR)
    

def ellipse_mask(img, mtbFlag=True):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    threshold = np.median(img_gray)
    diagThreshold = np.linalg.norm(img_gray.shape[:2])  * 0.7
    
    if mtbFlag:
        _, img_contours = cv2.threshold(img_gray, threshold, 255, 0)
    else:
        img_contours = cv2.Canny(img_gray, threshold, threshold +50)
    cv2.imshow('MTB' if mtbFlag else 'Canny', img)

    contours, _ = cv2.findContours(img_contours, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the rotated rectangles and ellipses for each contour
    minRects = [None]*len(contours)
    minEllipses = [None]*len(contours)
    for i, c in enumerate(contours):
        minRects[i] = cv2.minAreaRect(c)
        if len(c) > 5 and get_diagonal(minRects[i]) > diagThreshold:
            minEllipses[i] = cv2.fitEllipse(c)
        
    mask = np.zeros((img_contours.shape[:2]), dtype=np.uint8)
    ellipseDiags = [get_diagonal(minRect) if minRect else 0 for minRect in minEllipses]
    maxEllipseLoc = np.argmax(ellipseDiags)
    cv2.ellipse(mask, minEllipses[maxEllipseLoc], 255, -1)
    diskImg = cv2.bitwise_and(img_gray, mask)
    
    cv2.imshow('MTB' if mtbFlag else 'Canny', img_contours)
    cv2.imshow('Disk', diskImg)
    
    if cv2.waitKey(0) == 27:
        return
    
    
def get_rect_idxs_by_yolo(shape: np.array, rect: [float]):
    '''
    Переводит `rect` из формы `[rxc ryc rw rh]` в форму `[rowS:rowF colS:colF]`
    '''
    h, w = shape[:2]
    
    # относительные x, y, w, h
    rxc, ryc, rw, rh = rect
    rw *= 1.1
    rh *= 1.1
    
    def r2a(r, full): return int(r*full)
    
    # абсолютные x, y, w, h
    axc, ayc, aw, ah = r2a(rxc, w), r2a(ryc, h), r2a(rw, w), r2a(rh, h)
    
    rowS = ayc - ah//2 ; rowS = rowS if rowS >= 0 else 0
    rowF = rowS + ah   ; rowF = rowF if rowF < h else h-1
    colS = axc - aw//2 ; colS = colS if colS >= 0 else 0
    colF = colS + aw   ; colF = colF if colF < w else w-1
    
    return rowS, rowF, colS, colF
    

def get_disks_by_yrects(img: cv2.Mat, yoloRects: [float]):
    '''
    Извлекает диски по координатам ограничивающих прямоугольников в формате YOLO
    '''
    # yoloRect == [xc yc w h]
    shape = img.shape
    disks = []
    for yoloRect in yoloRects:
        rs, rf, cs, cf = get_rect_idxs_by_yolo(shape, yoloRect)
        disks.append(img[rs:rf, cs:cf])
    
    return disks

def get_disks_by_yolo(name: str, datasetPath: str):
    '''
    Извлекает диски, найденные YOLO \n
    Параметр `name` должен быть без расширения!
    '''
    img = cv2.imread(f'{datasetPath}\\images\\{name}.jpg', -1)
    yoloRects = []
    with open(f'{datasetPath}\\labels\\{name}.txt', 'r') as f:
        for line in f:
            if line[0] == '0':
                yoloRect = [float(a) for a in line[2:].split()]
                if len(yoloRect) > 4:
                    pass
                yoloRects.append(yoloRect)
    return get_disks_by_yrects(img, yoloRects)

def imwrite_disks(datasetDir: str):
    names = [name[:-4] for name in os.listdir(f'{datasetDir}/labels')]
    disks = []
    diskNames = []
    maskDir = Dataset.DISK_MASKS
    for name in names:
        for i, disk in enumerate(get_disks_by_yolo(name, datasetDir)):
            disks.append(disk)
            diskNames.append(f'{name}_{i}')
            cv2.imwrite(f'{maskDir}/{name}_{i}.jpg', disk)

def get_disks():
    disks = []
    maskDir = Dataset.DISK_MASKS
    names = os.listdir(maskDir)
    for name in names:
        disk = cv2.imread(os_join(maskDir, name))
        disks.append(disk)
            
    return disks, names
            
def make_dbscan_masks(disks: [cv2.Mat], maskDir: str):
    if not os.path.exists(maskDir):
        os.mkdir(maskDir)
        
    for i, disk in enumerate(disks):
        # приводим изображение к формату датасета
        datasetImg =  make_dataset_from_img(disk, xyFlag=False)
        # кластеризация
        dbLabels, dbDiskImg = db_clustering(datasetImg, disk)
        
        if i < 30:
            cv2.imshow('Disk (H)', disk)
            cv2.imshow('Disk masked', dbDiskImg)
            # cv2.imshow('Mask', dbLabels)
            
            if cv2.waitKey(0) == 27:
                cv2.destroyAllWindows()
                
def make_em_masks(disks: [cv2.Mat], maskDir: str):
    if not os.path.exists(maskDir):
        os.mkdir(maskDir)
        
    for i, disk in enumerate(disks):
        # приводим изображение к формату датасета
        datasetImg =  make_dataset_from_img(disk, xyFlag=False)
        # кластеризация
        emLabels, emDiskImg = em_clustering(datasetImg, disk)
        
        if i < 30:
            cv2.imshow('Disk (H)', disk)
            cv2.imshow('Disk masked', emDiskImg)
            emLabels = np.array(np.reshape(emLabels * 50, disk.shape[:2]), np.uint8)
            
            cv2.imshow('Clusters', emLabels)
            
            if cv2.waitKey(0) == 27:
                cv2.destroyAllWindows()
                
def make_bfs_masks(disks: [cv2.Mat], names: [str], maskDir: str):
    if not os.path.exists(maskDir):
        os.mkdir(maskDir)
        
    for i, (disk, name) in enumerate(zip(disks, names)):
        diskImg = extract_disk_by_inner_fill(disk)
        # diskImg = extract_disk_by_outer_fill(disk)
        cv2.imwrite(f'{maskDir}/{name}.jpg', diskImg)
        
        if 0 < i < 60:
            cv2.imshow('Disk (H)', disk)
            cv2.imshow('Disk masked', diskImg)
            hsv = cv2.cvtColor(disk, cv2.COLOR_BGR2HSV)
            cv2.imshow('Sat', hsv[:, :, 1])
            canny =  cv2.Canny(hsv[:, :, 1], 200, 220)
            cv2.imshow('Can', cv2.GaussianBlur(canny, (7, 7), 0)) #cv2.GaussianBlur(canny, (5, 5), 0)
            
            if cv2.waitKey(0) == 27:
                cv2.destroyAllWindows()
        
def get_hog(img):
    # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # img = hsv[:, :, 1]
    img = np.float32(img) / 255.0
    
    # Calculate gradient
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)
    
    # Calculate gradient magnitude and direction ( in degrees )
    mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    
    # idxs = np.stack(np.where(mag > np.max(mag)/2), axis=-1)
    # img = np.zeros_like(mag)
    # img[idxs[:, 0], idxs[:, 1]] = np.ones_like(idxs[:, 0])*255
    idxs = np.where(mag > np.max(mag)/10)
    img = np.zeros_like(mag)
    img[idxs] = np.ones_like(idxs[0])*255
    # img[*idxs] = np.ones_like(idxs[0])
    
    return np.uint8(img)
    
def get_contours(img: cv2.Mat):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    s = hsv[:,:,1]
    _, thresh = cv2.threshold(s, np.median(s), 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)    
    # print(contours.shape)
    contoursImg = np.zeros(s.shape)
    cv2.drawContours(contoursImg, contours, -1, (255), 2)
    
    return contoursImg

def get_contours_by_hog(hog: cv2.Mat):
    merged = hog[:,:, 0]
    for i in range(1, 3):
        merged = np.bitwise_or(merged, hog[:,:,i])
        
    # _, thresh = cv2.threshold(merged, np.median(merged), 255, cv2.THRESH_BINARY)
    scale = 1
    shape = np.array(merged.shape[::-1], np.float32)
    merged = cv2.resize(merged, np.array((shape * scale), np.uint8))
    
    contours, _ = cv2.findContours(merged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)    
    contoursImg = np.zeros(merged.shape)
    
    contours = sorted(contours, key=lambda x: len(x), reverse=True)
    for i in range(min(10, len(contours))):
        cv2.drawContours(contoursImg, contours, i, (255), 1)
    # cv2.drawContours(contoursImg, contours, -1, (255), 1)
    print(contours[20])
    contoursImg = cv2.resize(contoursImg, np.array(shape, np.uint8))
    
    contoursImg = np.array(contoursImg, np.uint8)
    
    return contoursImg

def sse(points: np.array):
    # y = ax + b
    x = points[:, 0]
    y = points[:, 1]
    
    
    xyMean = np.mean(x*y)
    xMean = np.mean(x)
    yMean = np.mean(y)
    x2Mean = np.mean(x**2)
    xMean2 = np.mean(x)**2
    
    a = (xyMean - xMean*yMean) / (x2Mean - xMean2)
    b = yMean - a*xMean
    # b = 0
    
    return a, b

def get_ab(diskMask: cv2.Mat):
    points = np.stack(np.where(diskMask > 100)[1::-1], axis=-1)
    a, b = sse(points)
    return a, b

def get_angle(a):
    ''' y = `a`x + b '''
    angle = np.arctan(a) * 180/np.pi
    return angle
    
def plot_regression(diskMask:cv2.Mat, ab):
    a, b = ab
    def f(x):
        return int(a*x + b)
    
    if len(diskMask.shape) < 3:
        diskMask = cv2.cvtColor(diskMask, cv2.COLOR_GRAY2BGR)
    
    x = 0
    pt1 = (x, f(x))
    x = diskMask.shape[1]-1
    pt2 = (x, f(x))
    cv2.line(diskMask, pt1, pt2, color=(0, 0, 255), thickness=2)
    
    cv2.imshow('SSE', diskMask)
    
    
def get_axis_of_mask(diskMask: cv2.Mat) -> np.ndarray:
    if len(diskMask.shape) != 2: # если канал не один
        diskMask = cv2.cvtColor(diskMask, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(diskMask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cont = max(contours, key=lambda x: len(x))
    def cast(num, base):
        mod = num % base
        return int(num + (-mod if mod <= base/2 else base-mod))
    
    if len(cont) > 5:
        ellipseRect = cv2.fitEllipse(cont)       
        (cx, cy), (w, h), alphaZ_ = ellipseRect
        
        alphaZ = cast(alphaZ_, 18)
        alphaZs = (270-alphaZ, 90-alphaZ if 90-alphaZ >= 0 else 450-alphaZ)
        
        ratio = w/h
        alphaX = cast(ratio*90-15, 15)
        alphaXs = (360+alphaX, 360-alphaX)  
        
        print(alphaXs, alphaZs)
        
    return (alphaXs, alphaZ)
    


def draw_ellipse_on_mask(diskMask: cv2.Mat) -> cv2.Mat:
    
    contours, _ = cv2.findContours(diskMask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(diskMask.shape) < 3:
        diskMask = cv2.cvtColor(diskMask, cv2.COLOR_GRAY2BGR)
    
    ellipses = []
    cont = max(contours, key=lambda x: len(x))
    # for i, cont in enumerate(contours):
    if len(cont) > 5:
        ellipses.append(cv2.fitEllipse(cont))
        
        alpha = ellipses[-1][-1]
        b = ellipses[-1][0][1]
        c = ellipses[-1][0][0]
        
        def f(x):
            return int(np.tan(np.deg2rad(alpha+90))*(x-c) + b)
        
        x1 = 0
        x2 = diskMask.shape[1]-1
        
        cv2.ellipse(diskMask, ellipses[-1], (0,0,255), thickness=2)
        cv2.line(diskMask, (x1,f(x1)), (x2, f(x2)), (255,0,0), 2)
        
    # cv2.imshow('ellipse', diskMask)
    return diskMask
    

def test_disk_extracting():
    # Папка с изображениями
    image_folder = './details'
    
    # Получаем список всех .jpg файлов в папке
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]
    
    for image_file in image_files:
        # Формируем полный путь к изображению
        image_path = os.path.join(image_folder, image_file)
        
        # Загружаем изображение
        img = cv2.imread(image_path, -1)
        
        # Проверяем, удалось ли загрузить изображение
        if img is None:
            print(f"Ошибка при загрузке изображения: {image_path}")
            continue
        
        # Применяем ваши функции обработки
        print(1)
        hog = get_hog(img)
        print(2)
        diskMask = extract_disk_by_inner_fill(img)
        print(3)
        contours = get_contours_by_hog(hog)
        print(4)
        
        # Отображаем результаты
        cv2.imshow('Disk (H)', img)
        cv2.imshow('HOG (H)', hog)
        cv2.imshow('Contours', contours)
        plot_regression(diskMask, get_ab(diskMask))
        
        # Закрытие окон по нажатию ESC
        if cv2.waitKey(0) == 27:
            cv2.destroyAllWindows()
            
def get_masks():
    maskNames = []
    masks = []
    for maskName in os.listdir(Dir.TEMPLATES):
        if maskName[-4:] == '.png' and int(maskName[:3]) >= 360:
            mask = cv2.imread(f'{Dir.TEMPLATES}/{maskName}', cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (320, 320))
            
            maskNames.append(maskName)
            masks.append(mask)
            
    return maskNames, np.array(masks)
            
def test_sse_on_templates():
    for name, mask in zip(*get_masks()):
        # cv2.imshow('Mask', mask)
        plot_regression(mask, get_ab(mask))
        
        if cv2.waitKey(0) == 27:
            cv2.destroyAllWindows()
            
def test_ellipses_on_templates():
    for name, mask in zip(*get_masks()):
        # cv2.imshow('Mask', mask)
        get_axis_of_mask(mask)
        mask = draw_ellipse_on_mask(mask)
        
        cv2.imshow(name, mask)
        if cv2.waitKey(0) == 27:
            cv2.destroyAllWindows()
            
def test_ellipses_on_real_data():
    disks, names = get_disks(Dataset.VIETNAM_DISKS_HB_t)
    for disk, name in zip(disks, names):
        mask = extract_disk_by_inner_fill(disk)
        get_axis_of_mask(mask)
        mask = draw_ellipse_on_mask(mask)
        # cv2.imshow('Mask', mask)
        
        cv2.imshow(name, mask)
        if cv2.waitKey(0) == 27:
            cv2.destroyAllWindows()
            
# def test_ellipses_on_real_data():
#     disks, names = get_disks(Dataset.VIETNAM_DISKS_HB_t)
#     for disk in disks:
#         mask = extract_disk_by_inner_fill(disk)
#         mask = draw_ellipse_on_mask(mask)
        
#         cv2.imshow('ellipse', mask)
#         if cv2.waitKey(0) == 27:
#             cv2.destroyAllWindows()
            
def test_sse():
    n = 200
    a = 5
    x = np.linspace(0, 10, n)
    noise = 10
    y1 = a*x + np.ones_like(x)*3 + np.random.rand(n)*noise - np.ones_like(x)*noise/2
    x_ = np.linspace(2, 8, n)
    y1_ = a*x_ - np.ones_like(x_)*3 + np.random.rand(n)*noise - np.ones_like(x_)*noise/2
    
    x = np.stack((*x, *x_), axis=0)
    y1 = np.stack((*y1, *y1_), axis=0)
    
    a, b = sse(np.stack((x, y1), axis=-1))
    
    y2 = a*x + np.ones_like(x)*b
    
    plt.plot(x, y1, '.', x, y2)
    plt.show()
            
if __name__ == '__main__':
    # test_sse()
    test_disk_extracting()
    # test_sse_on_templates()
    # test_ellipses_on_templates()
    # test_ellipses_on_real_data()
    
