import glob
import os
import random

import numpy as np
import torch
import torchvision.transforms as transforms
from matplotlib import pyplot as plt

from CustomDataset import CustomDataset
from CustomLoss import CustomSparseCategoricalCrossentropy
from ShorelineNetModel import unet_model

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset, ConcatDataset, TensorDataset

from utils.fgsm import load_attacked_image


def split_dataset(dataset, num_splits=3):
    length = len(dataset)
    indices = list(range(length))
    split_size = length // num_splits
    subsets = [Subset(dataset, indices[i * split_size: (i + 1) * split_size]) for i in range(num_splits)]

    # 如果有余数，将多出的部分分给最后一个子集
    if length % num_splits != 0:
        subsets[-1] = Subset(dataset, indices[(num_splits - 1) * split_size:])

    return subsets


# Hyperparameters
N_CLASSES = 3
TRAIN_VAL_RATIO = 0.9
SEED = 42
BATCH_SIZE = 32
IMG_SIZE = 224

num_epochs = 30

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Train transformer
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),  # Resize the image to 224x224
])

# Load the standard data
std_images_path = os.path.join(os.getcwd(), 'data/images/')

# 获取所有图像文件的列表
all_std_images = glob.glob(std_images_path + '*.jpg')
random.shuffle(all_std_images)
# 按照比例划分训练集和验证集
train_std_images, _ = train_test_split(all_std_images, train_size=TRAIN_VAL_RATIO, random_state=SEED)

# Create the standard dataset
std_dataset = CustomDataset(train_std_images, transform=train_transform)
# 1. Split into three for three black-box attack
subsets = split_dataset(std_dataset)
# 2. To dataloader
dataloader1 = DataLoader(subsets[0], batch_size=BATCH_SIZE, shuffle=True)
dataloader2 = DataLoader(subsets[1], batch_size=BATCH_SIZE, shuffle=True)
dataloader3 = DataLoader(subsets[2], batch_size=BATCH_SIZE, shuffle=True)
# 3. To attakced dataset
_, image_tensor1, mask_tensor1 = load_attacked_image(dataloader1, device, method="Occlusion")
_, image_tensor2, mask_tensor2= load_attacked_image(dataloader2, device, method="Occlusion")
_, image_tensor3, mask_tensor3 = load_attacked_image(dataloader3, device, method="Rotation")
# 4. Merge
tmp_inputs = torch.cat((image_tensor1, image_tensor2), axis=0)
conbined_inputs = torch.cat((tmp_inputs, image_tensor3), axis=0)
tmp_masks = torch.cat((mask_tensor1, mask_tensor2))
combined_masks = torch.cat((tmp_masks, mask_tensor3), axis=0)
combined_dataset = TensorDataset(conbined_inputs, combined_masks)

# 5. Create the data augmentation dataloader
data_aug_dataset = ConcatDataset([combined_dataset, std_dataset])
data_aug_dataloader = DataLoader(combined_dataset, batch_size=BATCH_SIZE, shuffle=True)

print(len(data_aug_dataloader))


# Load the data for attacked image
images_path = os.path.join(os.getcwd(), 'data/attacked_images/')

# 获取所有图像文件的列表
all_images = glob.glob(images_path + '*.jpg')
random.shuffle(all_images)
# 按照比例划分训练集和验证集
train_images, _ = train_test_split(all_images, train_size=TRAIN_VAL_RATIO, random_state=SEED)

# Create the attacked dataset
train_dataset = CustomDataset(train_images, transform=train_transform, attack_allowed=True)

# Create the dataloader
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Initialize the model
model = unet_model(N_CLASSES)
# model.load_state_dict(torch.load('result/model.pt'))

# Move model to GPU if available
model.to(device)
model.train()


# Training Process
criterion = CustomSparseCategoricalCrossentropy()
optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)
print("Using device:", device)

# Save loss and accuracy for each epoch
train_loss_history = []
train_acc_history = []
for epoch in range(num_epochs):
    print(f'Epoch {epoch + 1}/{num_epochs}')
    print('-' * 10)

    model.train()

    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in data_aug_dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        # print('Pred', outputs.shape)
        # print('True', labels.squeeze(1).long(), labels.squeeze(1).long().shape)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels.squeeze(1).data)

    epoch_loss = running_loss / len(data_aug_dataloader.dataset)
    epoch_acc = running_corrects.double() / (len(data_aug_dataloader.dataset) * 224 * 224)

    print(f'Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
    train_loss_history.append(epoch_loss)
    train_acc_history.append(epoch_acc.cpu().item())

torch.save(model.state_dict(), 'result/model_augment.pt')

# Train use the white-box attacked image
for epoch in range(20):
    print(f'Epoch {epoch + 1}/{20}')
    print('-' * 10)

    model.train()

    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels.squeeze(1).data)

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = running_corrects.double() / (len(train_loader.dataset) * 224 * 224)

    print(f'Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
    train_loss_history.append(epoch_loss)
    train_acc_history.append(epoch_acc.cpu().item())

torch.save(model.state_dict(), 'result/model_attacked_optimized.pt')


# 绘制损失值和准确率的图像
epochs = range(1, 51)

plt.figure(figsize=(12, 5))

# 绘制损失值曲线
plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss_history, '-')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

# 绘制准确率曲线
plt.subplot(1, 2, 2)
plt.plot(epochs, train_acc_history, '-')
plt.title('Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

# 显示图像
plt.tight_layout()
plt.show()


