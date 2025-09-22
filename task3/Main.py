import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import pandas as pd
import numpy as np
from PIL import Image
import cv2
import os
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ==================== КОНФИГУРАЦИЯ ====================
class Config:
    # Пути к данным
    train_csv = "train3/train.csv"
    test_csv = "train3/test.csv"
    sample_submission = "train3/sample_submission.csv"
    image_dir = "train3//images/"  # предполагаемая структура папок
    
    # Параметры модели
    model_name = 'efficientnet_b0'
    pretrained = True
    num_classes = 1
    
    # Обучение
    batch_size = 32
    num_epochs = 50
    learning_rate = 1e-4
    weight_decay = 1e-5
    
    # Аугментации
    image_size = 224
    
    # TTA
    tta_transforms = 8
    
    # Устройство
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = Config()

# ==================== ДАТАСЕТ И АУГМЕНТАЦИИ ====================
def clahe_transform(image):
    """Применение CLAHE для улучшения контраста ночных изображений"""
    image_np = np.array(image)
    
    # Конвертация в LAB цветовое пространство
    lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # Применение CLAHE к L-каналу
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    
    # Объединение обратно
    lab_clahe = cv2.merge([l_clahe, a, b])
    image_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
    
    return Image.fromarray(image_clahe)

class SkyQualityDataset(Dataset):
    def __init__(self, df, transform=None, is_train=True):
        self.df = df
        self.transform = transform
        self.is_train = is_train
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename = row['filename']
        
        # Загрузка изображения
        img_path = os.path.join(config.image_dir, filename)
        image = Image.open(img_path).convert('RGB')
        
        # Применение CLAHE для тренировочных данных
        if self.is_train:
            image = clahe_transform(image)
        
        if self.transform:
            image = self.transform(image)
        
        if self.is_train:
            target = torch.tensor(row['NSB_mpsas'], dtype=torch.float32)
            return image, target
        else:
            return image, filename

# Аугментации для тренировочных данных
train_transform = transforms.Compose([
    transforms.Resize((config.image_size, config.image_size)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Аугментации для валидационных данных
val_transform = transforms.Compose([
    transforms.Resize((config.image_size, config.image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# TTA трансформации
tta_transforms = []
for i in range(config.tta_transforms):
    if i == 0:
        # Оригинальное изображение
        tta_transforms.append(val_transform)
    elif i == 1:
        # Горизонтальное отражение
        tta_transforms.append(transforms.Compose([
            transforms.Resize((config.image_size, config.image_size)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]))
    else:
        # Комбинации аугментаций
        tta_transforms.append(transforms.Compose([
            transforms.Resize((config.image_size, config.image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]))

# ==================== МОДЕЛЬ ====================
class SkyQualityModel(nn.Module):
    def __init__(self, model_name='efficientnet_b0', pretrained=True, num_classes=1):
        super(SkyQualityModel, self).__init__()
        
        if model_name == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        
        self.regressor = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        return self.regressor(features)

# ==================== ФУНКЦИИ ОБУЧЕНИЯ ====================
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    
    for batch_idx, (images, targets) in enumerate(tqdm(dataloader, desc="Training")):
        images = images.to(device)
        targets = targets.to(device).unsqueeze(1)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    return running_loss / len(dataloader)

def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Validation"):
            images = images.to(device)
            targets = targets.to(device).unsqueeze(1)
            
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            all_preds.extend(outputs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
    return running_loss / len(dataloader), rmse

# ==================== TTA ПРЕДСКАЗАНИЕ ====================
def predict_tta(model, dataloader, device, tta_transforms):
    model.eval()
    all_preds = []
    filenames = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="TTA Prediction"):
            if len(batch) == 2:
                images, batch_filenames = batch
            else:
                images = batch
                batch_filenames = [f"img_{i}" for i in range(len(images))]
            
            batch_preds = []
            
            for transform in tta_transforms:
                # Применяем TTA трансформации
                transformed_images = torch.stack([transform(Image.fromarray((img.permute(1,2,0).numpy() * 255).astype(np.uint8))) 
                                                for img in images])
                
                transformed_images = transformed_images.to(device)
                outputs = model(transformed_images)
                batch_preds.append(outputs.cpu().numpy())
            
            # Усредняем предсказания по TTA
            batch_preds = np.mean(batch_preds, axis=0)
            all_preds.extend(batch_preds)
            filenames.extend(batch_filenames)
    
    return np.array(all_preds), filenames

# ==================== ОСНОВНАЯ ФУНКЦИЯ ====================
def main():
    print("Загрузка данных...")
    
    # Загрузка данных
    train_df = pd.read_csv(config.train_csv)
    test_df = pd.read_csv(config.test_csv)
    
    print(f"Размер тренировочных данных: {len(train_df)}")
    print(f"Размер тестовых данных: {len(test_df)}")
    
    # Разделение на тренировочную и валидационную выборки
    from sklearn.model_selection import train_test_split
    train_data, val_data = train_test_split(train_df, test_size=0.2, random_state=42)
    
    # Создание датасетов и даталоадеров
    train_dataset = SkyQualityDataset(train_data, transform=train_transform, is_train=True)
    val_dataset = SkyQualityDataset(val_data, transform=val_transform, is_train=False)
    test_dataset = SkyQualityDataset(test_df, transform=val_transform, is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)
    
    # Инициализация модели
    model = SkyQualityModel(config.model_name, config.pretrained, config.num_classes)
    model = model.to(config.device)
    
    # Функция потерь и оптимизатор
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    print("Начало обучения...")
    best_rmse = float('inf')
    best_model = None
    
    # Обучение
    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch+1}/{config.num_epochs}")
        
        # Тренировка
        train_loss = train_epoch(model, train_loader, criterion, optimizer, config.device)
        
        # Валидация
        val_loss, val_rmse = validate_epoch(model, val_loader, criterion, config.device)
        
        scheduler.step(val_loss)
        
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val RMSE: {val_rmse:.4f}")
        
        # Сохранение лучшей модели
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_model = model.state_dict().copy()
            torch.save(best_model, 'best_sky_quality_model.pth')
            print(f"Новая лучшая модель сохранена с RMSE: {best_rmse:.4f}")
    
    # Загрузка лучшей модели для предсказания
    model.load_state_dict(torch.load('best_sky_quality_model.pth'))
    print(f"Лучшая модель загружена с RMSE: {best_rmse:.4f}")
    
    # Предсказание на тестовых данных с TTA
    print("Предсказание на тестовых данных с TTA...")
    test_preds, filenames = predict_tta(model, test_loader, config.device, tta_transforms)
    
    # Калибровка предсказаний с помощью изотонической регрессии
    print("Калибровка предсказаний...")
    iso_reg = IsotonicRegression(out_of_bounds='clip')
    
    # Используем валидационные данные для калибровки
    val_preds, _ = predict_tta(model, val_loader, config.device, [val_transform])
    val_true = val_data['NSB_mpsas'].values
    
    # Обучение калибратора
    iso_reg.fit(val_preds.flatten(), val_true)
    
    # Применение калибровки к тестовым предсказаниям
    calibrated_preds = iso_reg.predict(test_preds.flatten())
    
    # Создание сабмита
    submission_df = pd.DataFrame({
        'filename': filenames,
        'NSB_mpsas': calibrated_preds
    })
    
    # Сопоставление с sample_submission
    sample_submission = pd.read_csv(config.sample_submission)
    final_submission = sample_submission[['idx']].copy()
    
    for idx in sample_submission['idx']:
        if idx in test_df['idx'].values:
            pred_value = submission_df[submission_df['filename'] == 
                                     test_df[test_df['idx'] == idx]['filename'].iloc[0]]['NSB_mpsas'].iloc[0]
            final_submission.loc[final_submission['idx'] == idx, 'NSB_mpsas'] = pred_value
    
    # Сохранение результатов
    final_submission.to_csv('train3/sky_quality_predictions.csv', index=False)
    print("Предсказания сохранены в sky_quality_predictions.csv")
    
    # Вычисление финального RMSE на валидации
    final_val_preds = iso_reg.predict(val_preds.flatten())
    final_rmse = np.sqrt(mean_squared_error(val_true, final_val_preds))
    print(f"Финальный RMSE после калибровки: {final_rmse:.4f}")
    
    # Расчет баллов
    points = 100 * max(0, 1 - final_rmse/0.30)
    print(f"Предварительные баллы: {points:.1f}")

if __name__ == "__main__":
    main()