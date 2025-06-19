import torch
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from tqdm import tqdm
import os
import random
import imagehash
import numpy as np
import time
import copy
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from torch.utils.data import DataLoader

def init_random_seed(value=42):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed(value)
    torch.backends.cudnn.benchmark = True

def find_and_visualize_duplicates(train_dir, val_dir):
    """Находит и визуализирует дубликаты между train и val"""
    def get_image_hash(img_path):
        try:
            with Image.open(img_path) as img:
                return imagehash.average_hash(img)
        except:
            return None

    # Собираем хеши
    train_hashes = {}
    for f in os.listdir(train_dir):
        h = get_image_hash(os.path.join(train_dir, f))
        if h is not None:
            train_hashes[h] = f

    val_hashes = {}
    duplicates = []
    for f in os.listdir(val_dir):
        h = get_image_hash(os.path.join(val_dir, f))
        if h is not None:
            val_hashes[h] = f
            if h in train_hashes:
                duplicates.append((train_hashes[h], f))

    # Визуализация
    if duplicates:
        print(f"\nНайдено {len(duplicates)} дубликатов:")
        for i, (train_file, val_file) in enumerate(duplicates[:5]):  # Показываем первые 5
            plt.figure(figsize=(10, 5))
            
            plt.subplot(1, 2, 1)
            img = Image.open(os.path.join(train_dir, train_file))
            plt.imshow(img)
            plt.title(f'Train: {train_file}')
            
            plt.subplot(1, 2, 2)
            img = Image.open(os.path.join(val_dir, val_file))
            plt.imshow(img)
            plt.title(f'Val: {val_file}')
            
            plt.show()
    else:
        print("\nДубликаты не найдены!")
    
    return duplicates

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.class_names = sorted(list(set(
            [f.split('_')[0] for f in os.listdir(root_dir)]
        )))
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        self.images = []
        
        for f in os.listdir(root_dir):
            class_name = f.split('_')[0]
            self.images.append((
                os.path.join(root_dir, f),
                self.class_to_idx[class_name]
            ))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

class AlexNetModified(nn.Module):
    def __init__(self, num_classes):
        super(AlexNetModified, self).__init__()
        self.model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
        self.model.classifier[6] = nn.Linear(self.model.classifier[6].in_features, num_classes)
    
    def forward(self, x):
        return self.model(x)

def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=30, patience=5):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs-1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in tqdm(dataloaders[phase], desc=phase):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            
            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc.cpu().numpy())

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val':
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    
                if epochs_no_improve >= patience:
                    print(f'Ранняя остановка на эпохе {epoch+1}!')
                    model.load_state_dict(best_model_wts)
                    return model, history

        scheduler.step(epoch_loss)

    model.load_state_dict(best_model_wts)
    return model, history

def plot_training_history(history):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Validation')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(model, dataloader, class_names):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', 
                xticklabels=class_names, 
                yticklabels=class_names,
                cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

    print(classification_report(all_labels, all_preds, target_names=class_names))

if __name__ == '__main__':
    init_random_seed()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset_dir = 'C:/Users/anast/Desktop/VKR/all_images'
    train_dir = os.path.join(dataset_dir, 'train')
    val_dir = os.path.join(dataset_dir, 'val')
    
    # Находим и показываем дубликаты
    duplicates = find_and_visualize_duplicates(train_dir, val_dir)
    
    if duplicates:
        print("Рекомендации:")
        print("1. Удалите файлы из val/ или переразделите данные")
        print("2. Можно автоматически удалить командой:")
        print("for dup in duplicates: os.remove(os.path.join(val_dir, dup[1]))")
        input("Нажмите Enter чтобы продолжить или Ctrl+C для прерывания...")

    # Аугментации
    data_transforms = {
        'train': transforms.Compose([
        transforms.Resize(256),  # Фиксированный размер
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.3),  # Уменьшите вероятность
        transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Только яркость/контраст
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    dataset_dir = 'C:/Users/anast/Desktop/VKR/all_images'
    
    # Проверка дубликатов между train и val
    train_files = set(os.listdir(os.path.join(dataset_dir, 'train')))
    val_files = set(os.listdir(os.path.join(dataset_dir, 'val')))
    common_files = train_files & val_files
    print(f"Общие файлы между train и val: {len(common_files)}")

    # Датсеты
    image_datasets = {
        'train': CustomDataset(os.path.join(dataset_dir, 'train'), data_transforms['train']),
        'val': CustomDataset(os.path.join(dataset_dir, 'val'), data_transforms['val'])
    }

    # Веса классов для несбалансированных данных
    class_counts = []
    for cls in image_datasets['train'].class_names:
        count = len([f for f in os.listdir(os.path.join(dataset_dir, 'train')) 
                   if f.startswith(cls)])
        class_counts.append(count)
    
    class_weights = 1. / torch.tensor(class_counts, dtype=torch.float32)
    class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=32, shuffle=True, num_workers=4),
        'val': DataLoader(image_datasets['val'], batch_size=32, shuffle=False, num_workers=4)
    }

    # Инициализация модели
    model = AlexNetModified(len(image_datasets['train'].class_names)).to(device)
    
    # Оптимизатор с L2-регуляризацией
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

    # Обучение модели
    model, history = train_model(
        model, dataloaders, criterion, optimizer, scheduler,
        num_epochs=60, patience=5
    )

    # Визуализация процесса обучения
    plot_training_history(history)
    
    # Оценка на валидационном наборе
    print("\nРезультаты на валидационных данных:")
    plot_confusion_matrix(model, dataloaders['val'], image_datasets['train'].class_names)

    # Сохранение модели
    input_tensor = torch.rand(1, 3, 224, 224)       
    script_model = torch.jit.trace(model.to(torch.device("cpu")), input_tensor)
    NAME = 'sight_recognizer_pretrained_999'
    script_model.save("C:/Users/anast/Desktop/VKR/"+NAME+".pt")

    """
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
    """