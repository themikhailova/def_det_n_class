# def_det_n_class
Данный проект создается для распознавания и классификации дефектов на различных [деталях](https://drive.google.com/drive/folders/1pdHhjAuZXqzPTgiYeGQCL0cEMqkWQ2TN?usp=sharing)

## Use case
![image](https://github.com/user-attachments/assets/317e29a9-b4a1-4b9b-b48d-06c9ba7edca6)


## Пример удаления фона
![image](https://github.com/user-attachments/assets/14129158-73c2-4003-804c-9874415e6d66)



## Пример выделения аномалии
![image](https://github.com/user-attachments/assets/9ae90e4b-6d54-4250-9d05-0a0391a0eee3)




## Файлы
/3d/toSixSides.py - создание изображений объекта эталона (3D-модели) с разных ракурсов – верхний, нижний, задний, передний, левый, правый ракурсы  
/preprocess/backremoveCV.py - удаление фона на основе цвета  
defectsdetector.py - выделение и детектирование аномалий на основе сравнения с изображением эталона  


