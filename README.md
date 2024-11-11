# def_det_n_class
Данный проект создается для распознавания и классификации дефектов на различных [деталях](https://drive.google.com/drive/folders/1pdHhjAuZXqzPTgiYeGQCL0cEMqkWQ2TN?usp=sharing)

## Use Case: Распознавание дефектов с учетом эталонных деталей
![image](https://github.com/user-attachments/assets/579208f4-3e2a-42c1-8e4f-7430fa185526)

**Цель:** Определить, является ли деталь идеальной или содержит дефекты

**Основной сценарий:**

1. **Человек** делает фотографию новой детали.
2. **Программа** получает фото, обрабатывает его, в т.ч. убирает фон и всевозможные лишние детали в окружении, классифицирует деталь на фото, в соответсвии с выявленным классом сравнивает фотографию с признаками эталонных изображений деталей деталей этого класса.
3. Если фотография соответствует эталону (проходит заданное пороговое значение), программа подтверждает, что деталь идеальна.
Если программа обнаруживает отклонения от эталона, она помечает деталь как "возможный брак". Дальше выделяет конкретный дефект и классифицирует его

**Ожидаемый результат:**
Программа автоматически идентифицирует детали, отклоняющиеся от эталона, и уточняет типы дефектов, накапливая данные для дальнейшего самообучения.

## Предобработка (удаление фона)
**1. Преобразование цветового пространства и выравнивание гистограммы**
Цветовое пространство LAB: работаем отдельно с яркостью (канал L) изображения для выравнивания яркости
Выравнивание гистограммы: равномерно распределяем значения яркости для улучшения различимости деталей относительно фона

**2. Сглаживание изображения (Гауссов фильтр)**
Гауссово размытие: уменьшаем шум и смягчаем изображение, чтобы убрать мелкие шумовые детали, не затрагивая основные структуры изображения

**3. Выделение текстурных признаков (Local Binary Patterns - LBP)**
LBP: выделяем структурные признаки для более четкого определения краев

**4. Сегментация изображения (методы GrabCut и Watershed)**

GrabCut: метод сегментации, основанный на построении графа, отделяет объекты от фона, используя начальную прямоугольную область, где находится объект.

Watershed: метод сегментации, разделяющий объекты на основе формы и глубины. Он создает "водораздел" по границам объектов, начиная от начальных маркеров, которые определяются с помощью морфологических операций и трансформации расстояния

Две полученные макси складываются в одну, т.к. GrabCut хорошо отделяет объект от основного фона, а Watershed лучше видит тень.

**Пример удаленного фона**

![image](https://github.com/user-attachments/assets/c396b6fb-539d-413d-a913-969e8d8c483d)

## Классификация 
Для классификации деталей был выбран **случайный лес** — ансамблевый метод, состоящий из множества деревьев решений, каждое из которых строится на случайной подвыборке данных. При классификации каждого изображения учитываются результаты всех деревьев, и на выходе используется наиболее вероятный класс
Для настройки модели применяется GridSearchCV с LeaveOneOut кросс-валидацией

## Feature matching
Для выяснения, является ли деталь приближенной к идеальной или имеет какие-либо отклонения, используется **feature matching**. Как и для классификации признаки деталей с фотографий извлекаются методом SIFT. 
Для каждой пары совпадающих дескрипторов алгоритм сопоставления возвращает два ближайших совпадения по расстоянию. Только те совпадения, у которых расстояние до первого ближайшего соседа существенно меньше, считаются хорошими совпадениями. Далее идет сравнение результата с порогом для окончательного определения, является ли деталь приближенной к идеальной или имеет дефекты. После фильтрации "хороших" совпадений вычисляется общее количество хороших совпадений, это число делится на общее количество дескрипторов в новом изображении, и получается соотношение совпадений, которое показывает, насколько хорошо новое изображение совпадает с эталоном
