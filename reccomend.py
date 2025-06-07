import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
from PIL import Image
import torch
import torchvision.transforms as transforms
from sklearn.decomposition import PCA 

class SightRecommender:
    def __init__(self, vector_db_path, model_path, device='cpu'):
        self.device = device
        self.load_vector_db(vector_db_path)
        self.load_model(model_path)

        test_input = torch.randn(1, 3, 224, 224).to(device)
        with torch.no_grad():
            test_output = self.model(test_input)
        self.model_output_dim = test_output.shape[1]

        

    def load_vector_db(self, path):
        with open(path, 'rb') as f:
            self.vector_db = pickle.load(f)
        
        self.features = self.vector_db['features']
        self.filenames = self.vector_db['filenames']
        self.class_names = self.vector_db['class_names']
        self.class_to_idx = self.vector_db['class_to_idx']
        self.features = self.features / np.linalg.norm(self.features, axis=1, keepdims=True)
    
    def load_model(self, model_path):
        self.model = torch.jit.load(model_path)
        self.model.to(self.device)
        self.model.eval()
    
    def transform(self, image):
        transform_pipeline = transforms.Compose([
                                    transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        tensor_image = transform_pipeline(image)
        return tensor_image
    
    def get_recommendations(self, image, top_k=3, exclude_class=None):
        # Проверяем тип изображения
        if isinstance(image, torch.Tensor):
            # Если изображение уже тензор - просто переносим на устройство
            image_tensor = image.to(self.device)
        else:
            # Если это PIL Image - применяем transform
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Получаем эмбеддинг
        with torch.no_grad():
            embedding = self.model(image_tensor).cpu().numpy()
        
        if self.model_output_dim != self.features.shape[1]:
            min_dim = min(self.model_output_dim, self.features.shape[1])
            self.pca = PCA(n_components=min_dim)
            combined = np.vstack([self.features, np.zeros((1, self.features.shape[1])) ])
            self.pca.fit(combined)
            self.features = self.pca.transform(self.features)

        embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
        
        # Ищем похожие достопримечательности
        distances = cosine_similarity(embedding, self.features)
        indices = np.argsort(distances[0])[::-1][:top_k + 1]
        #distances, indices = self.vector_db.search(embedding.cpu().numpy(), top_k + 1)
        
        # Формируем результат
        results = []
        for i, idx in enumerate(indices):
            class_name = self.class_names[idx]
            if class_name == exclude_class:
                continue  # Пропускаем исключенный класс
            if len(results) >= top_k:
                break
            results.append({
                'class_name': class_name,
                'distance': distances[0][idx]
            })
        
        return results

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    recommender = SightRecommender(
        vector_db_path='vector_database.pkl',
        model_path='sight_recognizer_pretrained_777.pt',
        device=device
    )

    test_image = Image.open('cityadministration_002.jpg').convert('RGB')
    recommendations = recommender.get_recommendations(test_image)
    print(recommendations)