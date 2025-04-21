import json
import os
import shutil
from PIL import Image

# Путь к директории с изображениями
image_dir = "C:/Users/anast/Desktop/VKR/new_photos"
# Путь к директории для сохранения вырезанных фрагментов
output_dir = "CroppedLandmarks" 
# Путь к директории для сохранения аугментированных изображений
aug_dir = "AugLandmarks"

# Создание директорий, если их нет
os.makedirs(output_dir, exist_ok=True)
os.makedirs(aug_dir, exist_ok=True)

def AugData(input_path, output_path):
    """
    Функция для аугментации изображений (отражение и поворот на 180 градусов).

    Args:
        input_path (str): Путь к директории с исходными изображениями.
        output_path (str): Путь к директории для сохранения аугментированных изображений.
    """
    for filename in os.listdir(input_path):
        if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            # Открываем изображение
            image_path = os.path.join(input_path, filename)
            try:
                image = Image.open(image_path)

                # Поворачиваем изображение на 180 градусов
                rotated_image = image.rotate(180)
                # Сохраняем повернутое изображение
                rotated_filename = f"{filename[:-4]}_rotated.jpg"
                rotated_image.save(os.path.join(output_path, rotated_filename))

                # Отзеркаливаем изображение
                flipped_image = image.transpose(Image.FLIP_LEFT_RIGHT)
                # Сохраняем отзеркаленное изображение
                flipped_filename = f"{filename[:-4]}_flipped.jpg"
                flipped_image.save(os.path.join(output_path, flipped_filename))

            except Exception as e:
                print(f"Ошибка при обработке изображения {filename}: {e}")

# Аугментация изображений
AugData(image_dir, aug_dir)

print("Аугментация завершена!")
