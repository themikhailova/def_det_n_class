# def_det_n_class
Данный проект создается для распознавания и классификации дефектов на различных [деталях](https://drive.google.com/drive/folders/1pdHhjAuZXqzPTgiYeGQCL0cEMqkWQ2TN?usp=sharing)

## Use Case: Распознавание дефектов с учетом эталонных деталей
![image](https://github.com/user-attachments/assets/8f18593b-4ceb-41c7-8b48-d11413b1e5ed)

**Цель:** Определить, является ли деталь идеальной или содержит дефекты
**Основной сценарий:**
**Человек** делает фотографию новой детали.
**Программа** получает фото, обрабатывает его, в т.ч. убирает фон и всевозможные лишние детали в окружении, классифицирует деталь на фото, в соответсвии с выявленным классом сравнивает фотографию с признаками эталонных изображений деталей деталей этого класса.
Если фотография соответствует эталону (проходит заданное пороговое значение), программа подтверждает, что деталь идеальна.
Если программа обнаруживает отклонения от эталона, она помечает деталь как "возможный брак". Дальше выделяет конкретный дефект и классифицирует его
**Ожидаемый результат:**
Программа автоматически идентифицирует детали, отклоняющиеся от эталона, и уточняет типы дефектов, накапливая данные для дальнейшего самообучения.

## Предобработка
**1. Преобразование цветового пространства и выравнивание гистограммы**
Цветовое пространство LAB: работаем отдельно с яркостью (канал L) изображения для выравнивания яркости
Выравнивание гистограммы: равномерно распределяем значения яркости для улучшения различимости деталей относительно фона

**2. Сглаживание изображения (Гауссов фильтр)**
Гауссово размытие: уменьшаем шум и смягчаем изображение, чтобы убрать мелкие шумовые детали, не затрагивая основные структуры изображения

**3. Выделение текстурных признаков (Local Binary Patterns - LBP)**
LBP: выделяем структурные признаки для более четкого определения краев

**4. Сегментация изображения (методы GrabCut и Watershed)**
GrabCut: интерактивный метод сегментации, основанный на построении графа, отделяет объекты от фона, используя начальную прямоугольную область, где находится объект
Watershed: метод сегментации, разделяющий объекты на основе формы и глубины. Он создает "водораздел" по границам объектов, начиная от начальных маркеров, которые определяются с помощью морфологических операций и трансформации расстояния

Две полученные макси складываются в одну, т.к. GrabCut хорошо отделяет объект от основного фона, а Watershed лучше видит тень.

