import glob
import os
import random

import torch
from torchvision import transforms

from CustomDataset import CustomDataset
from CustomLoss import CustomSparseCategoricalCrossentropy
from ShorelineNetModel import unet_model
from utils.basicUtils import display, create_mask, cal_accuracy

from sklearn.model_selection import train_test_split

from utils.attack_methods import fgsm_attack, pgd_attack, bim_attack, load_attacked_image, occlusions_attack, rotation_attack

if __name__ == '__main__':
    N_CLASSES = 3
    BATCH_SIZE = 32
    TRAIN_VAL_RATIO = 0.9
    SEED = 42
    IMG_SIZE = 224

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    visual_transform = transforms.Compose([
        transforms.Resize((384, 512)),
    ])

    # Train transformer
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),  # Resize the image to 224x224
    ])

    # Load the data
    data_dir = os.path.join(os.getcwd(), 'data/')
    images_path = os.path.join(os.getcwd(), 'data/images/')
    labels_path = os.path.join(os.getcwd(), 'data/masks/')

    # 获取所有图像文件的列表
    all_images = glob.glob(images_path + '*.jpg')
    # random.shuffle(all_images)
    # 按照比例划分训练集和验证集
    train_images, val_images = train_test_split(all_images, train_size=TRAIN_VAL_RATIO, random_state=SEED)
    # print(val_images)
    # 创建数据集实例
    train_dataset = CustomDataset(train_images, transform=train_transform)
    val_dataset = CustomDataset(val_images, transform=train_transform)

    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    criterion = CustomSparseCategoricalCrossentropy()

    # Load the plain model
    model = unet_model(N_CLASSES)
    model.load_state_dict(torch.load('result/model.pt'))
    model.to(device)
    model.eval()

    # Select one test image for visualization
    image1, mask1 = val_dataset[1]
    image1, mask1 = image1.unsqueeze(0).to(device), mask1.to(device)

    pre_mask1 = create_mask(model(image1)).type(torch.uint8)

    # Calculate the acc
    cal_accuracy(model, val_loader, criterion, device, "Normal")

    # Attack one image for visualization
    epsilon = 0.01 # Set the perturbation rate
    attack_method = "PGD"
    alpha = 0.001
    iteraton = 10
    delta = 2

    if attack_method == "FGSM":
        print(f"Attack using {attack_method} method")
        perturbed_image = fgsm_attack(model, criterion, image1, mask1, epsilon=epsilon) # FGSM Attack
    elif attack_method == "BIM":
        print(f"Attack using {attack_method} method")
        perturbed_image = bim_attack(model, criterion, image1, mask1, epsilon=epsilon)  # BIM Attack
    elif attack_method == "PGD":
        print(f"Attack using {attack_method} method")
        perturbed_image = pgd_attack(model, criterion, image1, mask1, epsilon=epsilon)  # PGD Attack
    elif attack_method == "Occlusion":
        print(f"Attack using {attack_method} method")
        perturbed_image = occlusions_attack(image1, delta=delta)  # Occlusion attack
    elif attack_method == "Rotation":
        print(f"Attack using {attack_method} method")
        perturbed_image, perturbed_mask = rotation_attack(image1, mask1)

    # Create the mask for the perturbed image
    perturbed_pred_mask = create_mask(model(perturbed_image)).type(torch.uint8)

    # Load the adv dataset
    perturbed_dataset, _, _ = load_attacked_image(val_loader, device, model=model, loss_fn=criterion, method=attack_method, epsilon=epsilon, alpha=alpha, iteration=iteraton, delta=delta)
    adv_dataloader = torch.utils.data.DataLoader(perturbed_dataset, batch_size=BATCH_SIZE, shuffle=False)
    cal_accuracy(model, adv_dataloader, criterion, device, "Attack", val_loader)



    # Load the adversarial defense model
    model_defense = unet_model(N_CLASSES)
    model_defense.load_state_dict(torch.load('result/model_attacked_optimized.pt'))
    model_defense.to(device)
    model_defense.eval()

    # Predict the new attacked image
    perturbed_pred_mask_defense = create_mask(model_defense(perturbed_image)).type(torch.uint8)

    # # Calculate the acc
    cal_accuracy(model_defense, adv_dataloader, criterion, device, "Defense", val_loader)
    # cal_accuracy(model_defense, val_loader, criterion, device, "Defense", val_loader)
    # Visualization
    if attack_method == "Rotation":
        display([perturbed_image.squeeze(0), perturbed_mask / 255, pre_mask1 / 255, perturbed_pred_mask / 255,
                 perturbed_pred_mask_defense / 255])
    else:
        display([perturbed_image.squeeze(0), mask1/255, pre_mask1/255, perturbed_pred_mask/255, perturbed_pred_mask_defense/255])
