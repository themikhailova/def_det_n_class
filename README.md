# def_det_n_class
Данный проект создается для распознавания и классификации дефектов на различных [деталях](https://drive.google.com/drive/folders/1pdHhjAuZXqzPTgiYeGQCL0cEMqkWQ2TN?usp=sharing)

## Use case
![image](https://github.com/user-attachments/assets/4b772e41-2570-4fcd-827f-5347776f7999)
![image](https://github.com/user-attachments/assets/bea04a1a-40c9-42d6-bc03-c763c98af2fe)


## Пример удаления фона
![image](https://github.com/user-attachments/assets/14129158-73c2-4003-804c-9874415e6d66)



## Пример выделения аномалии
![image](https://github.com/user-attachments/assets/9ae90e4b-6d54-4250-9d05-0a0391a0eee3)




## Файлы
/3d/toSixSides.py - создание изображений объекта эталона (3D-модели) с разных ракурсов – верхний, нижний, задний, передний, левый, правый ракурсы  
/preprocess/backremoveCV.py - удаление фона на основе цвета  
defectsdetector.py - выделение и детектирование аномалий на основе сравнения с изображением эталона  


