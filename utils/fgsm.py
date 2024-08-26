import random

import torch
import torchattacks
from torch.utils.data import TensorDataset


def fgsm_attack(model, loss_fn, images, labels, epsilon=0.01):
    print(images.shape)
    # model.eval()
    with torch.enable_grad():
        images.requires_grad = True

        outputs = model(images)

        loss = loss_fn(outputs, labels)

        model.zero_grad()
        loss.backward()

        data_grad = images.grad.data
        sign_data_grad = data_grad.sign()
        perturbed_image = images + epsilon * sign_data_grad

        perturbed_image = torch.clamp(perturbed_image, 0, 1)

    return perturbed_image


# def pgd_attack(model, images, labels, epsilon=0.02, alpha=0.001, steps=40):
#     atk = torchattacks.PGD(model, eps=epsilon, alpha=alpha, steps=steps)
#
#     # Normalize the label
#     labels = labels/255
#     labels = labels.long()
#
#     adv_images = atk(images, labels)
#     return adv_images


def pgd_attack(model, loss_fn, x, y, epsilon=0.01, alpha=0.001, num_iterations=10):
    # Add random noise
    delta = torch.rand_like(x) * 2 * epsilon - epsilon
    x_adv = x + delta
    x_adv = torch.clamp(x_adv, 0, 1)

    for i in range(num_iterations):
        # Calculate the gradient of adversarial samples
        x_adv.requires_grad = True
        outputs = model(x_adv)
        loss = loss_fn(outputs, y)
        model.zero_grad()
        loss.backward()
        gradient = x_adv.grad.data

        # Update the adversarial samples
        x_adv = x_adv + alpha * gradient.sign()

        # 执行投影操作，确保对抗样本在 x 的 epsilon 邻域内
        eta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x + eta, min=0, max=1).detach()
    return x_adv


# def bim_attack(model, images, labels, epsilon=0.02, alpha=0.001, steps=40):
#     atk = torchattacks.BIM(model, eps=epsilon, alpha=alpha, steps=steps)
#
#     # Normalize the label
#     labels = labels/255
#     labels = labels.long()
#
#     adv_images = atk(images, labels)
#     return adv_images


def bim_attack(model, loss_fn, x, y, epsilon=0.01, alpha=0.001, num_iterations=10):
    x_adv = x.clone()
    for i in range(num_iterations):
        # Calculate the gradient of adversarial samples
        x_adv.requires_grad = True
        outputs = model(x_adv)
        loss = loss_fn(outputs, y)
        model.zero_grad()
        loss.backward()
        gradient = x_adv.grad.data

        # Update the adversarial samples
        x_adv = x_adv + alpha * gradient.sign()

        # 执行裁剪操作，确保对抗样本在 x 的 epsilon 邻域内
        eta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x + eta, min=0, max=1).detach()
    return x_adv


def add_random_occlusions(images, delta=1):
    batch_size, channels, height, width = images.shape

    def add_white_circle(image, delta):
        # 计算圆的大小，基于图片大小和delta调整
        mean_radius = height // 10  # 平均大小是1/10张图片的大小
        radius = int(mean_radius * (1 + delta * (random.random() - 0.5)))  # 调整圆的大小
        # 随机确定圆心的位置
        x_center = random.randint(radius, width - radius)
        y_center = random.randint(radius, height - radius)

        # 使用 torch.meshgrid 创建网格坐标
        y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')
        # 计算与中心的距离
        dist_from_center = torch.sqrt((x - x_center) ** 2 + (y - y_center) ** 2)

        # 生成径向渐变遮挡的掩码
        mask = torch.clamp(1.0 - dist_from_center / radius, min=0.0).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        # 将掩码应用到图像上
        for c in range(channels):
            image[c] = image[c] * (1 - mask) + mask * 1.0  # 1.0表示白色，如果图像标准化到[0, 1]

        return image

    def add_black_dots(image, delta):
        # 计算黑色点的数量，基于delta调整
        num_dots = int(100 * delta)  # delta 越大，黑点越多
        for _ in range(num_dots):
            # 随机确定点的位置
            x = random.randint(0, width - 1)
            y = random.randint(0, height - 1)
            # 在所有通道上设置黑点，值设为0（黑色）
            image[:, y, x] = 0.0

        return image

    # 遍历所有图片，并添加随机遮挡
    processed_images = torch.empty_like(images)
    for i in range(batch_size):
        img = images[i]
        # 随机选择添加白色圆形遮挡或黑色点遮挡
        # if random.choice([True, False]):
        #     img = add_white_circle(img, delta)
        # else:
        #     img = add_black_dots(img, delta)
        img = add_white_circle(img, delta)
        processed_images[i] = img

    return processed_images


def load_attacked_image(model, dataloader, loss_fn, device, method="FGSM", epsilon=0.01, alpha=0.001, iteration=10, delta=1):
    # Create empty lists to hold the attacked images and corresponding labels
    all_adv_images = []
    all_labels = []

    # Iterate over the dataloader
    for images, labels in dataloader:
        labels = labels.squeeze(1)
        images, labels = images.to(device), labels.to(device)

        # Perform the attack
        if method == "FGSM":
            adv_images = fgsm_attack(model, loss_fn, images, labels, epsilon=epsilon)
        elif method == "BIM":
            # adv_images = bim_attack(model, images, labels, epsilon=epsilon)
            adv_images = bim_attack(model, loss_fn, images, labels, epsilon=epsilon, alpha=alpha, num_iterations=iteration)
        elif method == "PGD":
            # adv_images = pgd_attack(model, images, labels, epsilon=epsilon)
            adv_images = pgd_attack(model, loss_fn, images, labels, epsilon=epsilon, alpha=alpha, num_iterations=iteration)
        elif method == "Occlusion":
            adv_images = add_random_occlusions(images, delta=delta)
        # Append the attacked images and labels to the lists
        all_adv_images.append(adv_images)
        all_labels.append(labels)

    # Concatenate all tensors in the lists along the first dimension (batch dimension)
    all_adv_images_tensor = torch.cat(all_adv_images, dim=0)
    all_labels_tensor = torch.cat(all_labels, dim=0)

    # Create a new DataLoader from the attacked images and labels
    adv_dataset = TensorDataset(all_adv_images_tensor, all_labels_tensor)

    # Print the shape of the resulting tensor for debugging
    return adv_dataset
