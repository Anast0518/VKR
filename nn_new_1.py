import torch
import torchvision.transforms as transforms
from torchvision import datasets, models
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from tqdm import tqdm
import os
import random
import numpy
import pandas
import time
import copy
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from torchvision.models import AlexNet_Weights
from sklearn.metrics import classification_report

def init_random_seed(value=0):
    random.seed(value)
    numpy.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed(value)
    torch.backends.cudnn.benchmark = True

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform):
        self.root = root
        self.transform = transform
        self.filenames = os.listdir(self.root)
        self.class_names = class_names
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.filenames[idx])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # Получаем класс из имени файла
        class_name = self.filenames[idx].split('_')[0]
        
        # Проверяем, есть ли метка для данного изображения
        label = self.class_to_idx.get(class_name, None)  # Возвращает None, есgли класса нет

        return image, label  # Теперь метка может быть None

class AlexNetModified(nn.Module):
    def __init__(self, num_classes):
        super(AlexNetModified, self).__init__()
        self.model = models.alexnet(weights=AlexNet_Weights.DEFAULT)
        self.model.classifier[6] = nn.Linear(self.model.classifier[6].in_features, num_classes)
    # weights=None
    def forward(self, x):
        return self.model(x)

def imshow(images, labels, class_names):
    """Отображает изображения из батча с соответствующими метками."""
    images = images.numpy().transpose((0, 2, 3, 1))  # Изменяем размерность для отображения
    plt.figure(figsize=(12, 6))
    num_images = min(len(images), 8)  # Отображаем не более 8 изображений
    for i in range(num_images):
        ax = plt.subplot(2, 4, i + 1)  # Отображаем 8 изображений
        plt.imshow(images[i].clip(0, 1))  # Обеспечиваем, что значения находятся в допустимом диапазоне
        plt.title(class_names[labels[i]] if labels[i] is not None else "Нет метки")
        plt.axis('off')
    plt.show()

def train_model(model, criterion, optimizer, scheduler, num_epochs=30, patience=5):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    epochs_without_improvement = 0
    # Для хранения истории потерь
    train_loss_history = []
    val_loss_history = []

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        model.train()
        running_loss = 0.0
        running_corrects = 0
        for inputs, labels in tqdm(dataloaders['train'], leave=False):
            inputs, labels = inputs.to(device), labels.to(device)

            # Пропускаем изображения без меток
            if labels is None:
                print("Пропущено изображение без метки.")
                continue

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(torch.argmax(outputs, 1) == labels.data)

        epoch_loss = running_loss / len(dataloaders['train'].dataset)
        epoch_acc = running_corrects.double() / len(dataloaders['train'].dataset)
        train_loss_history.append(epoch_loss)
        print(f'Обучающая Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        model.eval()
        running_loss = 0.0
        running_corrects = 0
        with torch.no_grad():
            for inputs, labels in dataloaders['val']:
                inputs, labels = inputs.to(device), labels.to(device)

                # Пропускаем изображения без меток
                if labels is None:
                    print("Пропущено изображение без метки.")
                    continue

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(torch.argmax(outputs, 1) == labels.data)

        epoch_loss = running_loss / len(dataloaders['val'].dataset)
        epoch_acc = running_corrects.double() / len(dataloaders['val'].dataset)
        val_loss_history.append(epoch_loss)
        print(f'Проверочная Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        
        if epochs_without_improvement >= patience:
            print(f'Обучение завершено: не наблюдалось улучшения в течение {patience} эпох.')
            break

        scheduler.step(epoch_loss)
    # Построение графиков после обучения
    loss_function_graph(train_loss_history, val_loss_history)
    model.load_state_dict(best_model_wts)
    return model

def extract_features(model, dataloader, device):
    """Извлекает признаки из предпоследнего слоя сети для всех изображений"""
    model.eval()
    features = []
    filenames = []
    class_names = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader):
            inputs = inputs.to(device)
            # Получаем выходы до последнего слоя (предпоследний слой)
            outputs = model.model.features(inputs)
            outputs = model.model.avgpool(outputs)
            outputs = torch.flatten(outputs, 1)
            features.append(outputs.cpu().numpy())
            
            # Сохраняем информацию о файлах и классах
            batch_filenames = [dataloader.dataset.filenames[i] for i in range(inputs.size(0))]
            filenames.extend(batch_filenames)
            batch_class_names = [batch_filenames[i].split('_')[0] for i in range(len(batch_filenames))]
            class_names.extend(batch_class_names)
    
    features = numpy.vstack(features)
    return features, filenames, class_names

# Функция для построения графика функции потерь
def loss_function_graph(train_loss_history, val_loss_history):
    plt.figure(figsize= (10, 5))
    plt.plot(train_loss_history, label = 'Функция потерь на обучающем наборе')
    plt.plot(val_loss_history, label = 'Функция потерь на проверочном наборе')
    plt.title('График функции потерь')
    plt.xlabel('Эпохи')
    plt.ylabel('Потеря')
    plt.legend()
    plt.grid(True)
    plt.show()

# Функция, которая показывает пример изображения до и после аугментации
def show_augmentation_example(dataset, idx=0):
    original_img = Image.open(os.path.join(dataset.root, dataset.filenames[idx])).convert('RGB')
    # Получаем аугментированное изображение
    augmented_img = dataset.transform(original_img) if dataset.transform else original_img
    # Преобразуем тензор обратно в изображение для отображения
    if isinstance(augmented_img, torch.Tensor):
        augmented_img = augmented_img.numpy().transpose((1, 2, 0))
        augmented_img = numpy.clip(augmented_img, 0, 1)
    # Визуализация
    plt.figure(figsize= (10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(original_img)
    plt.title('Оригинальное изображение')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(augmented_img)
    plt.title('Изображение после аугментации')
    plt.axis('off')
    plt.show()


def plot_confusion_matrix(model, dataloader, class_names):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)  # Переносим метки на device
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Проверяем, что нет пропущенных меток
    print("Примеры меток:", all_labels[:10])
    print("Примеры предсказаний:", all_preds[:10])
    
    # Убедимся, что метки соответствуют class_names
    print("Уникальные метки в all_labels:", numpy.unique(all_labels))
    
    cm = confusion_matrix(all_labels, all_preds)
    print("Матрица ошибок (сырая):\n", cm)
    
    cm_df = pandas.DataFrame(cm, index=class_names, columns=class_names)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
    plt.title('Матрица ошибок')
    plt.ylabel('Истинные классы')
    plt.xlabel('Предсказанные классы')
    plt.show()
    
    # Выводим отчёт о классификации
    print(classification_report(all_labels, all_preds, target_names=class_names))

if __name__ == '__main__':
    init_random_seed()

    dataset_name = 'all_images'
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            # Случайное изменение масштаба
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.RandomRotation(degrees=(-15, 15)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    dataset_dir = 'C:/Users/anast/Desktop/VKR/' + dataset_name
    class_names = []
    train_dir = os.path.join(dataset_dir, 'train')
    if os.path.isdir(train_dir):
        for filename in os.listdir(train_dir):
            class_name = filename.split('_')[0] 
            if class_name not in class_names:
                class_names.append(class_name)

    image_datasets = {
        'train': CustomDataset(os.path.join(dataset_dir, 'train'), data_transforms['train']),
        'val': CustomDataset(os.path.join(dataset_dir, 'val'), data_transforms['val'])
    }

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=16, shuffle=True, num_workers=4)
                  for x in ['train', 'val']}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Показываем пример аугментации перед обучением
    show_augmentation_example(image_datasets['train'])

    model = AlexNetModified(len(class_names)).to(device)
    print(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Уменьшена скорость обучения
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=3)

    # Пример отображения батча изображений
    inputs, labels = next(iter(dataloaders['train']))
    imshow(inputs, labels, class_names)

    model = train_model(model, criterion, optimizer, scheduler, num_epochs=60, patience=5)

    # Построение матрицы ошибок после обучения
    plot_confusion_matrix(model, dataloaders['val'], class_names)

    # Сохранение модели
    input_tensor = torch.rand(1, 3, 224, 224)       
    script_model = torch.jit.trace(model.to(torch.device("cpu")), input_tensor)
    NAME = 'sight_recognizer_pretrained_888'
    script_model.save("C:/Users/anast/Desktop/VKR/"+NAME+".pt")

    print("Создание базы векторных представлений...")
    all_features, all_filenames, all_class_names = extract_features(model, dataloaders['val'], device)
    
    vector_db = {
        'features': all_features,
        'filenames': all_filenames,
        'class_names': all_class_names,
        'class_to_idx': image_datasets['train'].class_to_idx
    }
    
    with open('vector_database.pkl', 'wb') as f:
        pickle.dump(vector_db, f)
    print("База векторов успешно сохранена!")
 