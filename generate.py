import os

import torch
import torchvision.transforms as transforms

from CustomLoss import CustomSparseCategoricalCrossentropy
from ShorelineNetModel import unet_model
from utils.basicUtils import parse_image
from utils.fgsm import fgsm_attack

if __name__ == '__main__':
    # Define some hyperparameters
    N_CLASSES = 3
    epsilon = 0.01
    IMG_SIZE = 224

    # Define the folder path
    # images_folder = 'data/test_image' # Only for test purpose
    images_folder = 'data/images'
    attacked_images_folder = 'data/attacked_images'
    os.makedirs(attacked_images_folder, exist_ok=True)

    # Load the trained model
    model = unet_model(N_CLASSES)
    model.load_state_dict(torch.load('result/model.pt'))

    # Load the loss function
    criterion = CustomSparseCategoricalCrossentropy()

    # 定义图像转换
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),  # Resize the image to 224x224
    ])

    # 定义逆转换
    inverse_transform = transforms.Compose([
        transforms.Resize((384, 512)),
        transforms.ToPILImage(),
    ])


    for image_name in os.listdir(images_folder):
        if image_name.endswith('.jpg'):
            # 读取图片
            image_path = os.path.join(images_folder, image_name)
            sample = parse_image(image_path)  # 使用之前定义的parse_image函数
            image, mask = sample['image'], sample['mask']
            image = train_transform(image).unsqueeze(0)
            mask = train_transform(mask).unsqueeze(0)
            mask = torch.round(mask).long()  # 四舍五入并转换为整数
            attacked_image = fgsm_attack(model, criterion, image, mask, epsilon)

            # 转换为PIL图片
            attacked_image = attacked_image.squeeze(0)  # 移除批次维度
            attacked_image = inverse_transform(attacked_image)

            # Save the attacked image
            attacked_image_path = os.path.join(attacked_images_folder, image_name)
            attacked_image.save(attacked_image_path)

    print("All images have been processed and saved.")

