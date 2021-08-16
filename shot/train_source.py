import logging
import os
import random
import sys

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
os.chdir(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from backbones import get_model
from networks import network
from networks.loss import CrossEntropyLabelSmooth
from configs.configs import Config
from dataloader import get_test_dataloader, get_source_dataloader, get_RMFD_source_dataloader
from utils.utils import Scheduler, topk_accuracy, cal_acc, cal_acc_oda, AverageMeter

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer


def init_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # cudnn.benchmark = True


def init_dataset_name(cfg):
    specific_name_dict = {
        "office-home": ['Art', 'Clipart', 'Product', 'RealWorld'],
        "RMFD": ['AFDB_face_dataset', 'AFDB_masked_face_dataset']
    }

    cfg.dataset.name_src = specific_name_dict[cfg.dataset.name][cfg.dataset.s]
    specific_name_dict[cfg.dataset.name].remove(cfg.dataset.name_src)
    cfg.dataset.name_tar = ' '.join(specific_name_dict[cfg.dataset.name])


def train(epoch, train_loader, netF, netB, netC, criterion, optimizer, scheduler):
    losses = AverageMeter()
    top1 = AverageMeter()
    epoch_sum = epoch * len(train_loader)

    netF.train()
    netB.train()
    netC.train()
    for i, data in enumerate(tqdm(train_loader, total=len(train_loader), leave=False, dynamic_ncols=True)):
        optimizer.zero_grad()
        scheduler(optimizer, epoch_sum + i + 1)
        x, y = data[0].to(device), data[1].to(device)

        y_prob = netC(netB(netF(x)))
        loss = criterion(y_prob, y)

        accuracy = topk_accuracy(y_prob, y, topk=(1,))
        losses.update(loss.item(), y_prob.size(0))
        top1.update(accuracy.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    log_str = f'Train: Accuracy = {100 - top1.avg:.2f}% | loss = {losses.avg:.3f}'
    return log_str


def validate(valid_loader, netF, netB, netC, cfg):
    netF.eval()
    netB.eval()
    netC.eval()
    if cfg.dataset.name == 'VISDA-C':
        acc_s_te, acc_list = cal_acc(valid_loader, netF, netB, netC, True)
        log_str = f'Valid: Accuracy = {acc_s_te:.2f}%\n{acc_list}'

    else:
        acc_s_te, mean_ent = cal_acc(valid_loader, netF, netB, netC, False)
        log_str = f'Valid: Accuracy = {acc_s_te:.2f}% | Mean_ent = {mean_ent:.3f}'

    return acc_s_te, log_str


def inference(test_loader, best_netF, best_netB, best_netC, target_name, cfg):
    if cfg.da.type == 'oda':
        acc_os1, acc_os2, acc_unknown = cal_acc_oda(test_loader, best_netF, best_netB, best_netC, cfg.train.epsilon,
                                                    cfg.dataset.num_class)
        log_str = f'Test: {cfg.dataset.name_src}=>{target_name} | Accuracy={acc_os2:.2f}% / {acc_os1:.2f}% / {acc_unknown:.2f}'
    else:
        if cfg.da.type == 'VISDA-C':
            acc, acc_list = cal_acc(test_loader, best_netF, best_netB, best_netC, True)
            log_str = f'Test: {cfg.dataset.name_src}=>{target_name} | Accuracy={acc:.2f}%\n{acc_list}'
        else:
            acc, _ = cal_acc(test_loader, best_netF, best_netB, best_netC, False)
            log_str = f'Test: {cfg.dataset.name_src}=>{target_name} | Accuracy={acc:.2f}%'

    logging.info(log_str)


@hydra.main(config_path='configs', config_name='config')
def main(cfg: Config) -> None:
    init_seed(cfg.train.seed)
    init_dataset_name(cfg)

    logging.info(OmegaConf.to_yaml(cfg))

    if cfg.dataset.name == "RMFD":
        train_loader, test_loader = get_RMFD_source_dataloader(cfg)
        valid_loader = test_loader
        test_loader_list = [test_loader]
    else:
        train_loader, valid_loader = get_source_dataloader(cfg)
        test_loader_list = get_test_dataloader(cfg)

    if cfg.train.use_pretrained_backbone:
        netF = get_model('r50', dropout=0.0, num_features=512).to(device)
        netF.load_state_dict(torch.load('../../data/backbone.pth'))
        netB = network.feat_bootleneck(512, cfg.model.bottleneck_dim, cfg.model.classifier).to(device)
    else:
        netF = network.ResBase('resnet50').to(device)
        netB = network.feat_bootleneck(netF.in_features, cfg.model.bottleneck_dim, cfg.model.classifier).to(device)
    netC = network.feat_classifier(cfg.dataset.num_class, cfg.model.bottleneck_dim, cfg.model.layer).to(device)

    params = [
        {"params": netF.parameters(), "lr": cfg.train.lr * 0.1},
        {"params": netB.parameters(), "lr": cfg.train.lr},
        {"params": netC.parameters(), "lr": cfg.train.lr},
    ]
    optimizer = torch.optim.SGD(params)
    optimizer = op_copy(optimizer)

    criterion = CrossEntropyLabelSmooth(cfg.dataset.num_class, cfg.model.epsilon)
    scheduler = Scheduler(cfg.train.max_epoch * len(train_loader))

    best_acc = 0
    best_netF = None
    best_netB = None
    best_netC = None
    interval = cfg.train.max_epoch // 10
    for epoch in range(cfg.train.max_epoch):
        train_log = train(epoch, train_loader, netF, netB, netC, criterion, optimizer, scheduler)

        if epoch % interval == 0 or epoch == cfg.train.max_epoch - 1:
            acc_val, val_log = validate(valid_loader, netF, netB, netC, cfg)

            if acc_val >= best_acc:
                best_acc = acc_val
                best_netF = netF.state_dict()
                best_netB = netB.state_dict()
                best_netC = netC.state_dict()

                torch.save(best_netF, "source_F.pt")
                torch.save(best_netB, "source_B.pt")
                torch.save(best_netC, "source_C.pt")

            logging.info(f"[{epoch + 1}/{cfg.train.max_epoch}] {train_log} {val_log}")
        else:
            logging.info(f"[{epoch + 1}/{cfg.train.max_epoch}] {train_log}")

    for test_loader, target_name in zip(test_loader_list, cfg.dataset.name_tar.split(' ')):
        netF.load_state_dict(best_netF)
        netB.load_state_dict(best_netB)
        netC.load_state_dict(best_netC)
        inference(test_loader, netF, netB, netC, target_name, cfg)


if __name__ == '__main__':
    main()
