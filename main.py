import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from PIL import UnidentifiedImageError
import matplotlib.pyplot as plt
import numpy as np
import os
import time

# 디바이스 설정
device = torch.device('xpu' if torch.xpu.is_available() else 'cpu')
print(f'Using device: {device}')

# ==================== 1. 데이터 전처리 및 로드 ====================
# 이미지 변환 정의
train_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 데이터 로드
data_dir = './data'
train_dir = os.path.join(data_dir, 'training_set')
test_dir = os.path.join(data_dir, 'test_set')


class SafeImageFolder(datasets.ImageFolder):
    """깨진 이미지를 건너뛰도록 보호된 ImageFolder"""

    def __getitem__(self, index):
        # 최대 len(samples)번까지 시도하여 깨진 이미지를 건너뛰기
        for _ in range(len(self.samples)):
            try:
                return super().__getitem__(index)
            except (UnidentifiedImageError, OSError):
                # 다음 인덱스로 이동
                index = (index + 1) % len(self.samples)
                continue
        # 전부 실패하면 예외 발생
        raise RuntimeError('All images appear to be corrupted.')


# ImageFolder를 사용하여 데이터셋 로드 (깨진 이미지 건너뛰기)
full_train_dataset = SafeImageFolder(root=train_dir, transform=train_transform)
test_dataset = SafeImageFolder(root=test_dir, transform=test_transform)

# 클래스 이름 확인
class_names = full_train_dataset.classes
print(f'Classes: {class_names}')
print(f'Total training images: {len(full_train_dataset)}')
print(f'Total test images: {len(test_dataset)}')

# Train/Validation 분할 (80/20)
train_size = int(0.8 * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size
train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

print(f'Training set size: {len(train_dataset)}')
print(f'Validation set size: {len(val_dataset)}')

# DataLoader 생성
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# ==================== 2. CNN 모델 정의 ====================
class CatDogCNN(nn.Module):
    def __init__(self):
        super(CatDogCNN, self).__init__()
        
        # Convolutional layers (deeper)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)

        
        # Pooling and activation
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        
        # Fully connected layers
        # 128x128 -> 64 -> 32 -> 16 -> 8 -> 4 (after 4 pooling layers)
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 2)
        
    def forward(self, x):
        # Conv block 1
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        # Conv block 2
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        # Conv block 3
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        # Conv block 4
        x = self.pool(self.relu(self.bn4(self.conv4(x))))
        
        # Flatten
        x = x.view(-1, 64 * 8 * 8)
        
        # Fully connected layers
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        
        return x
    
    def get_feature_maps(self, x):
        """각 conv layer의 feature map을 반환"""
        feature_maps = {}
        
        x = self.conv1(x)
        feature_maps['conv1'] = x.clone()
        x = self.pool(self.relu(self.bn1(x)))
        
        x = self.conv2(x)
        feature_maps['conv2'] = x.clone()
        x = self.pool(self.relu(self.bn2(x)))
        
        x = self.conv3(x)
        feature_maps['conv3'] = x.clone()
        x = self.pool(self.relu(self.bn3(x)))
        
        x = self.conv4(x)
        feature_maps['conv4'] = x.clone()
        x = self.pool(self.relu(self.bn4(x)))
        
        
        return feature_maps

# 모델 생성
model = CatDogCNN().to(device)
print(model)

# ==================== 3. 학습 설정 ====================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

# ==================== 4. 학습 함수 ====================
def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc

def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc

# ==================== 5. 학습 실행 ====================
num_epochs = 10
train_losses = []
val_losses = []
train_accs = []
val_accs = []

print("\n" + "="*60)
print("Training Started")
print("="*60)

for epoch in range(num_epochs):
    start_time = time.time()
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = validate(model, val_loader, criterion, device)
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)
    
    scheduler.step(val_loss)
    end_time = time.time()
      
    print(f'Epoch [{epoch+1}/{num_epochs}] | '
          f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | '
          f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% |' 
          f'time: {end_time - start_time}')

print("="*60)
print("Training Completed")
print("="*60)

# ==================== 6. 학습 결과 시각화 ====================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss 그래프
axes[0].plot(range(1, num_epochs+1), train_losses, 'b-', label='Train Loss')
axes[0].plot(range(1, num_epochs+1), val_losses, 'r-', label='Validation Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training and Validation Loss')
axes[0].legend()
axes[0].grid(True)

# Accuracy 그래프
axes[1].plot(range(1, num_epochs+1), train_accs, 'b-', label='Train Accuracy')
axes[1].plot(range(1, num_epochs+1), val_accs, 'r-', label='Validation Accuracy')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy (%)')
axes[1].set_title('Training and Validation Accuracy')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig('training_history.png', dpi=150)
plt.show()
print("Training history saved to 'training_history.png'")

# ==================== 7. 테스트 세트 평가 ====================
test_loss, test_acc = validate(model, test_loader, criterion, device)
print(f'\nTest Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.2f}%')

# ==================== 8. CNN 필터 시각화 ====================
def visualize_filters(model, layer_name='conv1'):
    """각 레이어의 필터 시각화"""
    # 모델에서 필터 가중치 추출
    if layer_name == 'conv1':
        filters = model.conv1.weight.data.cpu().clone()
    elif layer_name == 'conv2':
        filters = model.conv2.weight.data.cpu().clone()
    elif layer_name == 'conv3':
        filters = model.conv3.weight.data.cpu().clone()
    elif layer_name == 'conv4':
        filters = model.conv4.weight.data.cpu().clone()

    # 정규화
    filters = filters - filters.min()
    filters = filters / filters.max()
    
    # 시각화할 필터 수 결정
    n_filters = min(filters.shape[0], 32)
    n_cols = 8
    n_rows = (n_filters + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 2))
    fig.suptitle(f'{layer_name} Filters Visualization', fontsize=16)
    
    for idx in range(n_filters):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]
        
        if layer_name == 'conv1':
            # RGB 필터의 경우 3채널을 모두 표시
            filter_img = filters[idx].permute(1, 2, 0).numpy()
            ax.imshow(filter_img)
        else:
            # 다중 채널 필터의 경우 평균으로 표시
            filter_img = filters[idx].mean(dim=0).numpy()
            ax.imshow(filter_img, cmap='viridis')
        
        ax.set_title(f'Filter {idx+1}')
        ax.axis('off')
    
    # 빈 축 숨기기
    for idx in range(n_filters, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{layer_name}_filters.png', dpi=150)
    plt.show()
    print(f"Filter visualization saved to '{layer_name}_filters.png'")

print("\n" + "="*60)
print("Filter Visualization")
print("="*60)

# 각 레이어의 필터 시각화
visualize_filters(model, 'conv1')
visualize_filters(model, 'conv2')
visualize_filters(model, 'conv3')
visualize_filters(model, 'conv4')

# ==================== 9. Feature Map 시각화 ====================
def visualize_feature_maps(model, image_tensor, class_names, device):
    """각 레이어의 feature map 시각화"""
    model.eval()
    
    with torch.no_grad():
        # Feature map 추출
        feature_maps = model.get_feature_maps(image_tensor.to(device))
    
    # 원본 이미지 역정규화하여 표시
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    original_img = image_tensor.squeeze(0).cpu() * std + mean
    original_img = original_img.permute(1, 2, 0).numpy()
    original_img = np.clip(original_img, 0, 1)
    
    # 예측 결과
    with torch.no_grad():
        output = model(image_tensor.to(device))
        _, predicted = torch.max(output, 1)
        pred_class = class_names[predicted.item()]
    
    # 각 레이어의 feature map 시각화
    for layer_name, fmap in feature_maps.items():
        fmap = fmap.cpu().squeeze(0)  # (C, H, W)
        n_features = min(fmap.shape[0], 16)  # 최대 16개의 feature map 표시
        
        n_cols = 4
        n_rows = (n_features + n_cols) // n_cols  # +1 for original image
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, n_rows * 3))
        fig.suptitle(f'{layer_name} Feature Maps (Predicted: {pred_class})', fontsize=14)
        
        # 원본 이미지 표시
        axes[0, 0].imshow(original_img)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Feature maps 표시
        for idx in range(n_features):
            row = (idx + 1) // n_cols
            col = (idx + 1) % n_cols
            ax = axes[row, col]
            
            feature_map = fmap[idx].numpy()
            ax.imshow(feature_map, cmap='viridis')
            ax.set_title(f'Feature {idx+1}')
            ax.axis('off')
        
        # 빈 축 숨기기
        for idx in range(n_features + 1, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{layer_name}_feature_maps.png', dpi=150)
        plt.show()
        print(f"Feature map visualization saved to '{layer_name}_feature_maps.png'")

print("\n" + "="*60)
print("Feature Map Visualization")
print("="*60)

# 테스트 이미지에서 샘플 가져오기
sample_images, sample_labels = next(iter(test_loader))

# 고양이 이미지와 개 이미지 각각 하나씩 시각화
cat_idx = None
dog_idx = None

for idx, label in enumerate(sample_labels):
    if label.item() == 0 and cat_idx is None:  # cats
        cat_idx = idx
    elif label.item() == 1 and dog_idx is None:  # dogs
        dog_idx = idx
    if cat_idx is not None and dog_idx is not None:
        break

print("\n--- Cat Image Feature Maps ---")
if cat_idx is not None:
    visualize_feature_maps(model, sample_images[cat_idx].unsqueeze(0), class_names, device)

print("\n--- Dog Image Feature Maps ---")
if dog_idx is not None:
    visualize_feature_maps(model, sample_images[dog_idx].unsqueeze(0), class_names, device)

# ==================== 10. Class Activation Mapping (Grad-CAM) ====================
def generate_cam(model, image_tensor, target_class=None, target_layer='conv4'):
    """Grad-CAM 계산"""
    model.eval()
    activations = []
    gradients = []

    target_module = getattr(model, target_layer)

    def forward_hook(_, __, output):
        activations.append(output.detach())

    def backward_hook(_, grad_input, grad_output):
        gradients.append(grad_output[0].detach())

    fh = target_module.register_forward_hook(forward_hook)
    bh = target_module.register_full_backward_hook(backward_hook)

    outputs = model(image_tensor.to(device))
    if target_class is None:
        target_class = outputs.argmax(dim=1).item()
    elif torch.is_tensor(target_class):
        target_class = target_class.item()

    score = outputs[0, target_class]
    model.zero_grad()
    score.backward()

    # Grad-CAM 계산
    grads = gradients[-1][0]  # (C, H, W)
    acts = activations[-1][0]  # (C, H, W)
    weights = grads.mean(dim=(1, 2), keepdim=True)
    cam = (weights * acts).sum(dim=0)  # (H, W)
    cam = torch.relu(cam)
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)

    fh.remove()
    bh.remove()

    probs = torch.softmax(outputs, dim=1)[0].detach().cpu()
    return cam.cpu().numpy(), probs


def visualize_cam(model, image_tensor, class_names, device, target_layer='conv4'):
    cam, probs = generate_cam(model, image_tensor, target_class=None, target_layer=target_layer)
    pred_class = probs.argmax().item()

    # 원본 이미지 (역정규화)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    original_img = image_tensor.squeeze(0).cpu() * std + mean
    original_img = original_img.permute(1, 2, 0).numpy()
    original_img = np.clip(original_img, 0, 1)

    # CAM 리사이즈
    cam_resized = np.clip(cam, 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(f'Grad-CAM (Pred: {class_names[pred_class]} | probs: {probs.tolist()})', fontsize=12)

    axes[0].imshow(original_img)
    axes[0].set_title('Original')
    axes[0].axis('off')

    axes[1].imshow(cam_resized, cmap='jet')
    axes[1].set_title('CAM')
    axes[1].axis('off')

    axes[2].imshow(original_img)
    axes[2].imshow(cam_resized, cmap='jet', alpha=0.4)
    axes[2].set_title('Overlay')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig('grad_cam.png', dpi=150)
    plt.show()
    print("Grad-CAM saved to 'grad_cam.png'")


print("\n" + "="*60)
print("Grad-CAM")
print("="*60)

visualize_cam(model, sample_images[cat_idx if cat_idx is not None else 0].unsqueeze(0), class_names, device)

# ==================== 11. 모델 저장 ====================
torch.save(model.state_dict(), 'cat_dog_cnn_model.pth')
print("\nModel saved to 'cat_dog_cnn_model.pth'")

print("\n" + "="*60)
print("All tasks completed!")
print("="*60)