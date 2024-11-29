# def_det_n_class
Данный проект создается для распознавания и классификации дефектов на различных [деталях](https://drive.google.com/drive/folders/1pdHhjAuZXqzPTgiYeGQCL0cEMqkWQ2TN?usp=sharing)

## Use case
![image](https://github.com/user-attachments/assets/471b8813-e1eb-4570-af9c-207a7778b27f)

## Пример удаления фона
![image](https://github.com/user-attachments/assets/b6565a7d-8708-4ffd-b4b9-a778b9fe29ea) ![image](https://github.com/user-attachments/assets/650c81d3-d9ea-48ff-8f04-ef000cf7d38c)

## Пример выделения аномалии
![image](https://github.com/user-attachments/assets/67ab6c30-e64c-4e8c-9f40-de3114a10603) ![image](https://github.com/user-attachments/assets/0273fc47-8d7b-4f19-9442-d1a692d044f3)




## Файлы
/3d/toSixSides.py - создание изображений объекта эталона (3D-модели) с разных ракурсов – верхний, нижний, задний, передний, левый, правый ракурсы  
/preprocess/backremoveCV.py - удаление фона на основе цвета  
defectsdetector.py - выделение и детектирование аномалий на основе сравнения с изображением эталона  


