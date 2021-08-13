import os

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms


class ImageDataset(Dataset):
    def __init__(self, ds_list, is_source, num_src_class, num_tar_class, transform):
        super().__init__()
        self.transform = transform
        self.x = list()
        self.y = list()
        self.ds_list = ds_list

        if is_source:
            self.generate_source_dataset(num_src_class)
        else:
            self.generate_target_dataset(num_src_class, num_tar_class)

    def generate_source_dataset(self, num_src_class):
        for ds in self.ds_list:
            path = ds[0]
            label = ds[1]
            if label < num_src_class:
                self.x.append(path)
                self.y.append(label)

    def generate_target_dataset(self, num_src_class, num_tgt_class):
        for ds in self.ds_list:
            path = ds[0]
            label = ds[1]
            if label < num_tgt_class:
                label = ds[1] if ds[1] < num_src_class else num_src_class
                self.x.append(path)
                self.y.append(label)

    def __getitem__(self, idx):
        # img = cv2.imread(self.x[idx])
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = self.transform(image=img)['image']

        img = Image.open(self.x[idx]).convert('RGB')
        img = self.transform(img)

        y = torch.tensor(self.y[idx])

        return img, y

    def __len__(self):
        return len(self.x)


def get_source_dataloader(cfg, split_ratio=0.9):
    # train_transform = A.Compose([
    #     A.Resize(cfg.dataset.resize_size, cfg.dataset.resize_size),
    #     A.RandomCrop(cfg.dataset.crop_size, cfg.dataset.crop_size),
    #     A.HorizontalFlip(),
    #     A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #     ToTensorV2()
    # ])
    #
    # val_transform = A.Compose([
    #     A.Resize(cfg.dataset.resize_size, cfg.dataset.resize_size),
    #     A.CenterCrop(cfg.dataset.crop_size, cfg.dataset.crop_size),
    #     A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #     ToTensorV2()
    # ])

    train_transform = transforms.Compose([
        transforms.Resize((cfg.dataset.resize_size, cfg.dataset.resize_size)),
        transforms.RandomCrop(cfg.dataset.crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((cfg.dataset.resize_size, cfg.dataset.resize_size)),
        transforms.CenterCrop(cfg.dataset.crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    folder_dataset = ImageFolder(os.path.join(cfg.dataset.root, cfg.dataset.name, cfg.dataset.name_src),
                                 loader=lambda x: x)
    dataset_size = len(folder_dataset)
    split_length = int(dataset_size * split_ratio)
    train_list, valid_list = torch.utils.data.random_split(folder_dataset, [split_length, dataset_size - split_length])

    train_set = ImageDataset(train_list, True, cfg.dataset.src_class, cfg.dataset.tar_class, train_transform)
    valid_set = ImageDataset(valid_list, True, cfg.dataset.src_class, cfg.dataset.tar_class, val_transform)

    train_loader = DataLoader(train_set, batch_size=cfg.train.batch_size, shuffle=True, num_workers=cfg.train.worker)
    valid_loader = DataLoader(valid_set, batch_size=cfg.train.batch_size, shuffle=True, num_workers=cfg.train.worker)

    return train_loader, valid_loader


def get_target_dataloader(cfg):
    transform = transforms.Compose([
        transforms.Resize((cfg.dataset.resize_size, cfg.dataset.resize_size)),
        transforms.CenterCrop(cfg.dataset.crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    test_loader_list = list()

    for name_tar in cfg.dataset.name_tar.split(' '):
        folder_dataset = ImageFolder(os.path.join(cfg.dataset.root, cfg.dataset.name, name_tar), loader=lambda x: x)
        dataset = ImageDataset(folder_dataset, False, cfg.dataset.src_class, cfg.dataset.tar_class, transform)
        test_loader = DataLoader(dataset, batch_size=cfg.train.batch_size * 2, num_workers=cfg.train.worker,
                                 shuffle=True)

        test_loader_list.append(test_loader)

    return test_loader_list
