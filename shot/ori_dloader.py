import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder


def make_dataset(image_list, labels):
    if labels:
        len_ = len(image_list)
        images = [(image_list[i].strip(), labels[i, :]) for i in range(len_)]
    else:
        if len(image_list[0].split()) > 2:
            images = [(val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
        else:
            images = [(val.split()[0], int(val.split()[1])) for val in image_list]
    return images


def rgb_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def l_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('L')


class ImageList(Dataset):
    def __init__(self, image_list, labels=None, transform=None, target_transform=None, mode='RGB'):
        imgs = make_dataset(image_list, labels)

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)


def image_train(resize_size=256, crop_size=224, alexnet=False):
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


def image_test(resize_size=256, crop_size=224, alexnet=False):
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


src_classes = [i for i in range(65)]
tar_classes = [i for i in range(25)]


def data_load_source(cfg):
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = cfg.train.batch_size

    all_txt_src = ImageFolder(os.path.join(cfg.dataset.root, cfg.dataset.name, cfg.dataset.name_src),
                              loader=lambda x: x)

    txt_src = list()
    for x in all_txt_src:
        path = x[0]
        cls = x[1]
        txt_src.append(f"{path} {cls}")

    if not cfg.da.type == 'uda':
        label_map_s = {}
        for i in range(65):
            label_map_s[src_classes[i]] = i

        new_src = []
        for i in range(len(txt_src)):
            rec = txt_src[i]
            reci = rec.strip().split(' ')
            if int(reci[1]) in src_classes:
                line = reci[0] + ' ' + str(label_map_s[int(reci[1])]) + '\n'
                new_src.append(line)
        txt_src = new_src.copy()

    dsize = len(txt_src)
    tr_size = int(0.9 * dsize)
    tr_txt, te_txt = torch.utils.data.random_split(txt_src, [tr_size, dsize - tr_size])

    dsets["source_tr"] = ImageList(tr_txt, transform=image_train())
    dset_loaders["source_tr"] = DataLoader(dsets["source_tr"], batch_size=train_bs, shuffle=True,
                                           num_workers=4, drop_last=False)
    dsets["source_te"] = ImageList(te_txt, transform=image_test())
    dset_loaders["source_te"] = DataLoader(dsets["source_te"], batch_size=train_bs, shuffle=True,
                                           num_workers=4, drop_last=False)

    return dset_loaders["source_tr"], dset_loaders["source_te"]


def data_load_target(cfg):
    train_bs = cfg.train.batch_size
    loader_list = list()

    target_names = ['Clipart', 'Product', 'RealWorld']

    for name in target_names:
        all_txt_test = ImageFolder(os.path.join(cfg.dataset.root, cfg.dataset.name, name), loader=lambda x: x)

        txt_test = list()
        for x in all_txt_test:
            path = x[0]
            cls = x[1]
            txt_test.append(f"{path} {cls}")

        if not cfg.da.type == 'uda':
            label_map_s = {}
            for i in range(65):
                label_map_s[src_classes[i]] = i

            new_tar = []
            for i in range(len(txt_test)):
                rec = txt_test[i]
                reci = rec.strip().split(' ')
                if int(reci[1]) in tar_classes:
                    if int(reci[1]) in src_classes:
                        line = reci[0] + ' ' + str(label_map_s[int(reci[1])]) + '\n'
                        new_tar.append(line)
                    else:
                        line = reci[0] + ' ' + str(len(label_map_s)) + '\n'
                        new_tar.append(line)
            txt_test = new_tar.copy()

        ds = ImageList(txt_test, transform=image_test())
        dl = DataLoader(ds, batch_size=train_bs * 2, shuffle=True, num_workers=4, drop_last=False)

        loader_list.append(dl)

    return loader_list
