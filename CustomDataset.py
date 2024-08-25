import torch

from utils.basicUtils import parse_image


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, transform=None, attack_allowed=False):
        self.image_paths = image_paths
        self.transform = transform
        self.attack_allowed = attack_allowed

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        sample = parse_image(img_path, self.attack_allowed)  # 使用之前定义的parse_image函数
        image, mask = sample['image'], sample['mask']

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        mask = torch.round(mask).long()  # 四舍五入并转换为整数

        return image, mask