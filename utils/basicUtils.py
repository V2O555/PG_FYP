import re

import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms
from matplotlib import pyplot as plt


def parse_image(img_path, attack_allowed=False) -> dict:
    '''
    Loads the image and its mask, returns a dictionary

    Original      Returns
    0: obstacle   0: obstacle
    1: water      1: water
    2: sky        2: sky
    4: unknown    3: unknown

    Args
    ----------
    img_path : str
        Image filename pattern (glob)
        (Mask location is found using regex)
    attack_allowed : bool

    Returns
    ----------
    dict {'image': tensor (384 x 512 x 3), 'mask': tensor (384 x 512 x 1)}
        Dictionary mapping an image and its annotation.
    '''

    image = Image.open(img_path).convert("RGB")
    transform = transforms.ToTensor()
    image = transform(image)

    # For one Image path:
    # .../data/images/train/01_001.png
    # Its corresponding annotation path is:
    # .../data/annotations/train/01_001_label.png
    # mask_path = re.sub(r"test_image", "masks", img_path) # this line is used for test only
    if not attack_allowed:
        mask_path = re.sub(r"images", "masks", img_path)
    else:
        mask_path = re.sub(r"attacked_images", "masks", img_path)
    mask_path = re.sub(r".jpg", "m.png", mask_path)

    mask = Image.open(mask_path)
    # Convert to pytorch tensor
    mask = torch.from_numpy(np.array(mask)).unsqueeze(0)
    mask = torch.where(mask == 4, torch.tensor(3, dtype=torch.float32), mask)

    return {'image': image, 'mask': mask}

def display(display_list):
    '''
    Takes an array of images and plots them

    Args
    ----------
    display_list: list
      expecting [input, true mask, predicted mask]

    Returns
    ----------
    None, graph is displayed
    '''
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask',
             'Predicted Mask (Attacked Image)', 'Predicted Mask (Attacked Image) using defense model']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(transforms.ToPILImage()(display_list[i].cpu()))
        plt.axis('off')
    plt.show()


def create_mask(pred_mask):
    '''
    Finds top prediction mask and returns it
    '''
    pred_mask = torch.argmax(pred_mask, dim=1, keepdim=True)
    return pred_mask[0]  # Assuming batch size of 1, return the first element


def cal_accuracy(test_model, test_loader, criterion, device, model_type):
    test_model.eval()
    running_loss = 0.0
    running_corrects = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = test_model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.squeeze(1).data).double()

    epoch_loss = running_loss / len(test_loader.dataset)
    epoch_acc = running_corrects / (len(test_loader.dataset) * 244 * 244)

    print(f'{model_type} model: Loss: {epoch_loss:.4f} Acc: {epoch_acc * 100:.4f}%')
