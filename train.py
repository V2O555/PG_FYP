import glob
import os
import random

import torch
import torchvision.transforms as transforms
from matplotlib import pyplot as plt

from CustomDataset import CustomDataset
from CustomLoss import CustomSparseCategoricalCrossentropy
from ShorelineNetModel import unet_model

from sklearn.model_selection import train_test_split


# Hyperparameters
N_CLASSES = 3
TRAIN_VAL_RATIO = 0.9
SEED = 321
BATCH_SIZE = 32
IMG_SIZE = 224


num_epochs = 50

# Load the data
images_path = os.path.join(os.getcwd(), 'data/images/')
labels_path = os.path.join(os.getcwd(), 'data/masks/')

# 获取所有图像文件的列表
all_images = glob.glob(images_path + '*.jpg')
random.shuffle(all_images)
# 按照比例划分训练集和验证集
train_images, val_images = train_test_split(all_images, train_size=TRAIN_VAL_RATIO, random_state=SEED)

# Train transformer
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),  # Resize the image to 224x224
])

# Create the dataset
train_dataset = CustomDataset(train_images, transform=train_transform)
val_dataset = CustomDataset(val_images, transform=train_transform)

# Create the dataloader
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Initialize the model
model = unet_model(N_CLASSES)

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        # print('Pred', outputs.shape)
        # print('True', labels.squeeze(1).long().shape)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels.squeeze(1).data).double()

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = running_corrects / (len(train_loader.dataset) * 224 * 224)
    print(f'Loss: {epoch_loss:.4f} Acc: {epoch_acc * 100:.4f}%')

    train_loss_history.append(epoch_loss)
    train_acc_history.append(epoch_acc.cpu().item())

torch.save(model.state_dict(), 'result/model.pt')

# 绘制损失值和准确率的图像
epochs = range(1, num_epochs + 1)

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