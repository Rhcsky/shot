import os
from glob import glob

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms


class ImageDataset(Dataset):
    def __init__(self, ds_list, is_source, num_src_class, num_tar_class, transform, with_idx=False):
        super().__init__()
        self.transform = transform
        self.x = list()
        self.y = list()
        self.ds_list = ds_list
        self.with_idx = with_idx

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
        if self.with_idx:
            return img, y, idx
        else:
            return img, y

    def __len__(self):
        return len(self.x)


class FaceDataset(Dataset):
    def __init__(self, name_classes, root, transform, with_idx=False):
        self.transform = transform
        self.name_classes = name_classes
        self.root = root
        self.with_idx = with_idx

        self.x = list()
        self.y = list()

        self.init_dataset()

    def init_dataset(self):
        for idx, name in enumerate(self.name_classes):
            imgs = glob(os.path.join(self.root, name, '*'))
            if len(imgs) < 1:
                continue
            for path in imgs:
                self.x.append(path)
                self.y.append(idx)

    def __getitem__(self, idx):
        img = Image.open(self.x[idx]).convert('RGB')
        img = self.transform(img)

        y = torch.tensor(self.y[idx])
        if self.with_idx:
            return img, y, idx
        else:
            return img, y

    def __len__(self):
        return len(self.x)


def train_transform(resize_size, crop_size):
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


def test_transform(resize_size, crop_size):
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


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

    folder_dataset = ImageFolder(os.path.join(cfg.dataset.root, cfg.dataset.name, cfg.dataset.name_src),
                                 loader=lambda x: x)
    dataset_size = len(folder_dataset)
    split_length = int(dataset_size * split_ratio)
    train_list, valid_list = torch.utils.data.random_split(folder_dataset, [split_length, dataset_size - split_length])

    train_set = ImageDataset(train_list, True, cfg.dataset.src_class, cfg.dataset.tar_class,
                             train_transform(cfg.dataset.resize_size, cfg.dataset.crop_size))
    valid_set = ImageDataset(valid_list, True, cfg.dataset.src_class, cfg.dataset.tar_class,
                             test_transform(cfg.dataset.resize_size, cfg.dataset.crop_size))

    train_loader = DataLoader(train_set, batch_size=cfg.train.batch_size, shuffle=True, num_workers=cfg.train.worker)
    valid_loader = DataLoader(valid_set, batch_size=cfg.train.batch_size, shuffle=True, num_workers=cfg.train.worker)

    return train_loader, valid_loader


def get_test_dataloader(cfg):
    test_loader_list = list()

    for name_tar in cfg.dataset.name_tar.split(' '):
        folder_dataset = ImageFolder(os.path.join(cfg.dataset.root, cfg.dataset.name, name_tar), loader=lambda x: x)
        dataset = ImageDataset(folder_dataset, False, cfg.dataset.src_class, cfg.dataset.tar_class,
                               test_transform(cfg.dataset.resize_size, cfg.dataset.crop_size))
        test_loader = DataLoader(dataset, batch_size=cfg.train.batch_size * 2, num_workers=cfg.train.worker,
                                 shuffle=True)

        test_loader_list.append(test_loader)

    return test_loader_list


def get_target_dataloader(cfg):
    folder_dataset = ImageFolder(os.path.join(cfg.dataset.root, cfg.dataset.name, cfg.dataset.name_tar),
                                 loader=lambda x: x)

    target_dataset = ImageDataset(folder_dataset, False, cfg.dataset.src_class, cfg.dataset.tar_class,
                                  train_transform(cfg.dataset.resize_size, cfg.dataset.crop_size), with_idx=True)
    test_dataset = ImageDataset(folder_dataset, False, cfg.dataset.src_class, cfg.dataset.tar_class,
                                test_transform(cfg.dataset.resize_size, cfg.dataset.crop_size), with_idx=True)

    target_loader = DataLoader(target_dataset, batch_size=cfg.train.batch_size, num_workers=cfg.train.worker,
                               shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg.train.batch_size * 3, num_workers=cfg.train.worker,
                             shuffle=False)

    return target_loader, test_loader


def get_RMFD_source_dataloader(cfg):
    src_path = os.path.join(cfg.dataset.root, cfg.dataset.name, cfg.dataset.name_src)
    tar_path = os.path.join(cfg.dataset.root, cfg.dataset.name, cfg.dataset.name_tar)
    source_folder_dataset = ImageFolder(src_path, loader=lambda x: x)
    target_folder_dataset = ImageFolder(tar_path, loader=lambda x: x)
    same_classes = list(filter(lambda x: (x in source_folder_dataset.classes) and (x in target_folder_dataset.classes),
                               list(set(source_folder_dataset.classes + target_folder_dataset.classes))))

    source_dataset = FaceDataset(same_classes, src_path,
                                 train_transform(cfg.dataset.resize_size, cfg.dataset.crop_size))

    target_dataset = FaceDataset(same_classes, tar_path, test_transform(cfg.dataset.resize_size, cfg.dataset.crop_size))

    source_loader = DataLoader(source_dataset, batch_size=cfg.train.batch_size, num_workers=cfg.train.worker,
                               shuffle=True)

    target_loader = DataLoader(target_dataset, batch_size=cfg.train.batch_size * 3, num_workers=cfg.train.worker,
                               shuffle=False)

    return source_loader, target_loader


def get_RMFD_target_dataloader(cfg):
    src_path = os.path.join(cfg.dataset.root, cfg.dataset.name, cfg.dataset.name_src)
    tar_path = os.path.join(cfg.dataset.root, cfg.dataset.name, cfg.dataset.name_tar)
    source_folder_dataset = ImageFolder(src_path, loader=lambda x: x)
    target_folder_dataset = ImageFolder(tar_path, loader=lambda x: x)
    same_classes = list(filter(lambda x: (x in source_folder_dataset.classes) and (x in target_folder_dataset.classes),
                               list(set(source_folder_dataset.classes + target_folder_dataset.classes))))

    train_dataset = FaceDataset(same_classes, tar_path, train_transform(cfg.dataset.resize_size, cfg.dataset.crop_size))

    test_dataset = FaceDataset(same_classes, tar_path, test_transform(cfg.dataset.resize_size, cfg.dataset.crop_size))

    train_loader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, num_workers=cfg.train.worker,
                              shuffle=True)

    test_loader = DataLoader(test_dataset, batch_size=cfg.train.batch_size * 3, num_workers=cfg.train.worker,
                             shuffle=False)

    return train_loader, test_loader
