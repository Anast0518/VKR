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
import time
import copy
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.models import AlexNet_Weights

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

    model.load_state_dict(best_model_wts)
    return model

def test_model(model, criterion, dataloader, class_names):
    """Тестирование модели на тестовом наборе данных"""
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc='Testing'):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            _, preds = torch.max(outputs, 1)
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total_samples += inputs.size(0)
    
    test_loss = running_loss / total_samples
    test_acc = running_corrects.double() / total_samples
    
    print(f'\nTest Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}')
    return test_loss, test_acc

if __name__ == '__main__':
    init_random_seed()

    dataset_name = 'all_images'
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
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
        'test': transforms.Compose([  # Добавляем преобразования для теста
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
        'val': CustomDataset(os.path.join(dataset_dir, 'val'), data_transforms['val']),
        'test': CustomDataset(os.path.join(dataset_dir, 'test'), data_transforms['test'])  # Добавляем тестовый датасет
    }

    dataloaders = {
        'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=16, shuffle=True, num_workers=4),
        'val': torch.utils.data.DataLoader(image_datasets['val'], batch_size=16, shuffle=True, num_workers=4),
        'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=16, shuffle=False, num_workers=4)  # Для теста shuffle=False
    }

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = AlexNetModified(len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=3)

    # Обучение модели
    model = train_model(model, criterion, optimizer, scheduler, num_epochs=60, patience=5)

    # Тестирование модели на тестовом наборе
    print("\nTesting the model on test dataset...")
    test_loss, test_acc = test_model(model, criterion, dataloaders['test'], class_names)
    
    # Сохранение модели
    input_tensor = torch.rand(1, 3, 224, 224)
    script_model = torch.jit.trace(model.to(torch.device("cpu")), input_tensor)
    NAME = 'sight_recognizer_pretrained_1000'
    script_model.save("C:/Users/anast/Desktop/VKR/"+NAME+".pt")
    
    # Вывод результатов тестирования
    print(f"\nFinal Test Results:")
    print(f"Loss: {test_loss:.4f}")
    print(f"Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"Number of test samples: {len(image_datasets['test'])}")