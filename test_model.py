import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFile
import os
from tqdm import tqdm

# Разрешаем загрузку усеченных изображений
ImageFile.LOAD_TRUNCATED_IMAGES = True

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class TestDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.filenames = []
        self.class_names = []
        self.class_to_idx = {}
        
        # Безопасное чтение файлов
        for f in os.listdir(root_dir):
            try:
                img_path = os.path.join(root_dir, f)
                with Image.open(img_path) as img:
                    img.verify()  # Проверка целостности изображения
                self.filenames.append(f)
                class_name = f.split('_')[0]
                if class_name not in self.class_to_idx:
                    self.class_to_idx[class_name] = len(self.class_to_idx)
            except (IOError, OSError, Image.UnidentifiedImageError) as e:
                print(f"Skipping corrupted file {f}: {str(e)}")
        
        self.class_names = sorted(list(self.class_to_idx.keys()))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.filenames[idx])
        try:
            with Image.open(img_path) as img:
                image = img.convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            
            class_name = self.filenames[idx].split('_')[0]
            label = self.class_to_idx[class_name]
            return image, label
        except Exception as e:
            print(f"Error loading {img_path}: {str(e)}")
            # Возвращаем пустое изображение и флаг ошибки
            dummy_img = torch.zeros(3, 224, 224)
            return dummy_img, -1  # -1 будет означать ошибку

def test_model(model_path, test_dir):
    # Загружаем модель
    model = torch.jit.load(model_path)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Создаем тестовый датасет
    test_dataset = TestDataset(test_dir, transform=transform)
    
    # Фильтруем битые изображения
    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,  # Устанавливаем 0 для отладки
        pin_memory=True
    )
    
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)  # Игнорируем образцы с ошибками
    
    # Тестирование
    correct = 0
    total = 0
    total_loss = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Testing'):
            # Пропускаем битые изображения
            valid_mask = labels != -1
            if not valid_mask.any():
                continue
                
            inputs = inputs[valid_mask].to(device)
            labels = labels[valid_mask].to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            total_loss += loss.item() * inputs.size(0)
    
    if total == 0:
        print("No valid images found for testing!")
        return
    
    avg_loss = total_loss / total
    accuracy = correct / total
    
    print(f"\nTest Results:")
    print(f"Classes: {test_dataset.class_names}")
    print(f"Total valid test images: {total}")
    print(f"Average loss: {avg_loss:.4f}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

if __name__ == "__main__":
    model_path = "C:/Users/anast/Desktop/VKR/sight_recognizer_pretrained_999.pt"
    test_dir = "C:/Users/anast/Desktop/VKR/all_images/test"
    
    test_model(model_path, test_dir)