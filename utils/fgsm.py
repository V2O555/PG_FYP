import torch
import torchattacks
from torch.utils.data import TensorDataset


def fgsm_attack(model, loss_fn, images, labels, epsilon=0.01):
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


def load_attacked_image(model, dataloader, loss_fn, device, method="FGSM", epsilon=0.01, alpha=0.001, iteration=10):
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
