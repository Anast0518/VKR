# Скрипт для формирования датасета фотографий достопримечательностей

import random
import os

def split_images(input_path, ratio):
    # Создаем поддиректории train и val
    train_dir = os.path.join(input_path, 'train') # 80% картинок будет на train
    val_dir = os.path.join(input_path, 'val') # 20% картинок будет на val
    
    os.makedirs(train_dir, exist_ok= True)
    os.makedirs(val_dir, exist_ok= True)

    # Получаем список файлов с картинками
    image_files = [f for f in os.listdir(input_path) if f.lower().endswith(('.png', '.jpeg', '.jpg'))]

    # Перемешиваем файлы с картинками случайным образом
    random.shuffle(image_files)

    # Вычисляем количество изображений в каждой поддиректории
    image_count = int(len(image_files) * ratio)
    
    # Разделяем изображения на поддиректории
    for i, file in enumerate(image_files):
        if i < image_count:
            os.rename(os.path.join(input_path, file), os.path.join(train_dir, file))
        else:
            os.rename(os.path.join(input_path, file), os.path.join(val_dir, file))

ratio = 0.8 # 80% для обучения, 20% для валидации
image_path = 'C:/Users/anast/Desktop/VKR/all_images'
split_images(image_path, ratio)