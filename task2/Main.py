import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import pandas as pd
from PIL import Image
import os
from sklearn.metrics import f1_score
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Проверка доступности GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

class AnimalTracksDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, is_train=True):
        self.labels_df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.is_train = is_train
        
        # Создаем mapping для классов
        self.classes = ['Bear', 'Bird', 'Cat', 'Wolf', 'Leopard', 'Otter']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Проверяем структуру CSV
        print(f"CSV columns: {self.labels_df.columns.tolist()}")
        print(f"First few rows:\n{self.labels_df.head()}")
    
    def __len__(self):
        return len(self.labels_df)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Получаем имя файла
        row = self.labels_df.iloc[idx]
        filename = row['filename'] if 'filename' in row else row[0]
        
        img_name = os.path.join(self.root_dir, str(filename))
        
        try:
            image = Image.open(img_name).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_name}: {e}")
            # Возвращаем пустое изображение в случае ошибки
            image = Image.new('RGB', (224, 224), color='black')
        
        if self.is_train:
            # Получаем метку класса из one-hot encoding
            if 'Bear' in self.labels_df.columns:
                labels = row[['Bear', 'Bird', 'Cat', 'Wolf', 'Leopard', 'Otter']].values.astype(np.float32)
                label = np.argmax(labels)
            else:
                # Предполагаем, что метка в первом столбце после filename
                label = row[1] if len(row) > 1 else 0
            
            label = torch.tensor(label, dtype=torch.long)
        else:
            label = -1
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Трансформации для обучения
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Трансформации для валидации/теста
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def create_model(num_classes=6):
    model = models.resnet18(pretrained=True)
    
    # Замораживаем начальные слои
    for param in model.parameters():
        param.requires_grad = False
    
    # Размораживаем последние слои
    for param in model.layer4.parameters():
        param.requires_grad = True
    
    # Заменяем последний слой
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )
    
    return model

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    for images, labels in tqdm(dataloader, desc='Training'):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader)
    epoch_f1 = f1_score(all_labels, all_preds, average='macro')
    
    return epoch_loss, epoch_f1

def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Validation'):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader)
    epoch_f1 = f1_score(all_labels, all_preds, average='macro')
    
    return epoch_loss, epoch_f1

def main():
    # Параметры
    batch_size = 16
    num_epochs = 10
    learning_rate = 1e-3
    
    # Создаем датасеты и даталоадеры
    train_dataset = AnimalTracksDataset(
        csv_file='task2/train/_classes.csv',
        root_dir='task2/train/',
        transform=train_transform,
        is_train=True
    )
    
    print(f"Dataset size: {len(train_dataset)}")
    
    # Разделяем на train/val
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Создаем модель
    model = create_model(num_classes=6)
    model = model.to(device)
    
    # Оптимизатор и функция потерь
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Обучение
    best_f1 = 0.0
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 50)
        
        # Обучение
        train_loss, train_f1 = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Валидация
        val_loss, val_f1 = validate_epoch(model, val_loader, criterion, device)
        
        print(f'Train Loss: {train_loss:.4f} F1: {train_f1:.4f}')
        print(f'Val Loss: {val_loss:.4f} F1: {val_f1:.4f}')
        
        # Сохраняем лучшую модель
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), 'task2/best_model.pth')
            print('New best model saved!')
        
        print()
    
    print(f'Best Validation F1: {best_f1:.4f}')

def verify_submission_file(filename):
    """Проверяет корректность submission файла"""
    try:
        df = pd.read_csv(filename)
        print(f"\nVerification of {filename}:")
        print(f"Columns: {df.columns.tolist()}")
        print(f"Number of rows: {len(df)}")
        print(f"First 5 rows:\n{df.head()}")
        print(f"Unique labels: {sorted(df['label'].unique())}")
        
        # Проверяем формат
        expected_columns = ['id', 'label']
        if list(df.columns) != expected_columns:
            print(f"ERROR: Expected columns {expected_columns}, got {df.columns.tolist()}")
            return False
        
        print("✓ Submission file is valid")
        return True
        
    except Exception as e:
        print(f"Error verifying submission file: {e}")
        return False

def predict_test_set(model_path, test_dir, template_csv):
    """Создает submission файл на основе шаблона с именами файлов"""
    # Загружаем модель
    model = create_model(num_classes=6)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # Загружаем шаблон
    template_df = pd.read_csv(template_csv)
    print(f"Template columns: {template_df.columns.tolist()}")
    print(f"Number of images in template: {len(template_df)}")
    
    # Создаем трансформации для теста
    test_transform = val_transform
    
    predictions = []
    processed_files = []
    
    with torch.no_grad():
        for idx, row in tqdm(template_df.iterrows(), total=len(template_df), desc='Processing test images'):
            filename = row['filename']
            img_path = os.path.join(test_dir, filename)
            
            try:
                if os.path.exists(img_path):
                    image = Image.open(img_path).convert('RGB')
                    image = test_transform(image).unsqueeze(0).to(device)
                    
                    output = model(image)
                    _, pred = torch.max(output, 1)
                    
                    predictions.append(pred.item())
                    processed_files.append(filename)
                else:
                    print(f"File not found: {img_path}")
                    predictions.append(0)  # Default to Bear
                    processed_files.append(filename)
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                predictions.append(0)  # Default to Bear
                processed_files.append(filename)
    
    # Создаем submission файл в правильном формате
    result_df = pd.DataFrame({
        'id': range(len(predictions)),
        'label': predictions
    })
    
    # Сохраняем файл
    result_df.to_csv('task2/submission.csv', index=False)
    print(f'Submission file created with {len(result_df)} predictions')
    
    # Также сохраняем версию с именами файлов для отладки
    debug_df = pd.DataFrame({
        'id': range(len(predictions)),
        'filename': processed_files,
        'label': predictions
    })
    debug_df.to_csv('task2/submission_debug.csv', index=False)
    print('Debug file with filenames created: task2/submission_debug.csv')
    
    # Проверяем файл
    verify_submission_file('task2/submission.csv')
    
    return result_df

if __name__ == '__main__':
    # 1. Обучение модели
    print("Starting training...")
    #main()
    
    # 2. Предсказание на тестовых данных
    print("\nStarting prediction on test set...")
    submission = predict_test_set(
        'task2/best_model.pth', 
        'task2/test/',
        'path/to/your/template.csv'  # Укажите путь к вашему файлу с именами
    )
    
    print("\nPipeline completed successfully!")
    print("Final submission file: task2/result_task2_21_9_25.csv")