import logging
import os
import random
import sys

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.spatial.distance import cdist
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
os.chdir(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from configs.configs import Config
from dataloader import get_RMFD_target_dataloader, get_target_dataloader
from utils.utils import Scheduler, cal_acc, AverageMeter, Entropy
from networks import network

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def change_model_require_grad(model: torch.nn.Module, true_or_false: bool):
    for k, v in model.named_parameters():
        v.requires_grad = true_or_false


def obtain_label(loader, netF, netB, netC, cfg):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            feas = netB(netF(inputs))
            outputs = netC(feas)
            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = torch.nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)

    before_acc = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    if cfg.model.distance == 'cosine':
        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

    all_fea = all_fea.float().cpu().numpy()
    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()
    initc = aff.transpose().dot(all_fea)
    initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
    cls_count = np.eye(K)[predict].sum(axis=0)
    labelset = np.where(cls_count > cfg.train.threshold)
    labelset = labelset[0]

    dd = cdist(all_fea, initc[labelset], cfg.model.distance)
    pred_label = dd.argmin(axis=1)
    pred_label = labelset[pred_label]

    for _ in range(1):
        aff = np.eye(K)[pred_label]
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
        dd = cdist(all_fea, initc[labelset], cfg.model.distance)
        pred_label = dd.argmin(axis=1)
        pred_label = labelset[pred_label]

    after_acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)
    log_str = f'PseudoLabeling: Accuracy = {before_acc * 100:.2f}% -> {after_acc * 100:.2f}%'

    logging.info(log_str)

    return pred_label.astype('int')


def get_mem_label(cfg, netF, netB, netC, test_loader):
    netF.eval()
    netB.eval()
    mem_label = obtain_label(test_loader, netF, netB, netC, cfg)
    mem_label = torch.from_numpy(mem_label).cuda()

    return mem_label


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
    cfg.dataset.name_tar = specific_name_dict[cfg.dataset.name][cfg.dataset.t]

    if cfg.da.type == "pda":
        cfg.train.gent = False


def train(epoch, train_loader, netF, netB, netC, criterion, optimizer, scheduler, mem_label, cfg):
    losses = AverageMeter()
    epoch_sum = epoch * len(train_loader)
    netF.train()
    netB.train()
    for i, data in enumerate(tqdm(train_loader, total=len(train_loader), leave=False, dynamic_ncols=True)):
        optimizer.zero_grad()
        scheduler(optimizer, epoch_sum + i + 1)
        x, tar_idx = data[0].to(device), data[2].to(device)

        features_test = netB(netF(x))
        outputs_test = netC(features_test)

        if cfg.train.cls_par > 0:
            pred = mem_label[tar_idx]
            loss = criterion(outputs_test, pred)
            loss *= cfg.train.cls_par
        else:
            loss = torch.tensor(0.0).to(device)

        if cfg.train.ent:
            softmax_out = torch.nn.Softmax(dim=1)(outputs_test)
            entropy_loss = torch.mean(Entropy(softmax_out))
            if cfg.train.gent:
                msoftmax = softmax_out.mean(dim=0)
                gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + cfg.model.epsilon))
                entropy_loss -= gentropy_loss
            im_loss = entropy_loss * cfg.train.ent_par
            loss += im_loss

        losses.update(loss, x.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    log_str = f'Train: Loss = {losses.avg:.3f}'
    return log_str


def validate(valid_loader, netF, netB, netC, cfg):
    netF.eval()
    netB.eval()
    if cfg.dataset.name == 'VISDA-C':
        acc_s_te, acc_list = cal_acc(valid_loader, netF, netB, netC, True)
        log_str = f'Valid: Accuracy = {acc_s_te:.2f}%\n{acc_list}'

    else:
        acc_s_te, mean_ent = cal_acc(valid_loader, netF, netB, netC, False)
        log_str = f'Valid: Accuracy = {acc_s_te:.2f}% | Mean_ent = {mean_ent:.3f}'

    return acc_s_te, log_str


@hydra.main(config_path='configs', config_name='config')
def main(cfg: Config) -> None:
    init_seed(cfg.train.seed)
    init_dataset_name(cfg)

    logging.info(OmegaConf.to_yaml(cfg))

    if cfg.da.type == 'pda':
        cfg.train.gent = False

    if cfg.dataset.name == "RMFD":
        train_loader, test_loader = get_RMFD_target_dataloader(cfg)
    else:
        train_loader, test_loader = get_target_dataloader(cfg)

    netF = network.ResBase('resnet50').to(device)
    netB = network.feat_bootleneck(netF.in_features, cfg.model.bottleneck_dim, cfg.model.classifier).to(device)
    netC = network.feat_classifier(cfg.dataset.num_class, cfg.model.bottleneck_dim, cfg.model.layer).to(device)

    netF = torch.nn.parallel.DataParallel(netF)
    netB = torch.nn.parallel.DataParallel(netB)
    netC = torch.nn.parallel.DataParallel(netC)

    netC.eval()
    change_model_require_grad(netC, False)

    params = list()

    if cfg.train.lr_decay1 > 0:
        params.append({"params": netF.parameters(), "lr": cfg.train.lr * cfg.train.lr_decay1})
    else:
        change_model_require_grad(netF, False)

    if cfg.train.lr_decay2 > 0:
        params.append({"params": netB.parameters(), "lr": cfg.train.lr * cfg.train.lr_decay2})
    else:
        change_model_require_grad(netB, False)

    optimizer = torch.optim.SGD(params)
    optimizer = op_copy(optimizer)

    criterion = torch.nn.CrossEntropyLoss()
    scheduler = Scheduler(cfg.train.max_epoch * len(train_loader))

    best_acc = 0
    best_netF = None
    best_netB = None
    best_netC = None
    mem_label = None
    interval = cfg.train.max_epoch // 15
    for epoch in range(cfg.train.max_epoch):

        if epoch % interval == 0 and cfg.train.cls_par > 0:
            mem_label = get_mem_label(cfg, netF, netB, netC, test_loader)

        train_log = train(epoch, train_loader, netF, netB, netC, criterion, optimizer, scheduler, mem_label, cfg)

        if epoch % interval == 0 or epoch == cfg.train.max_epoch:
            acc_val, val_log = validate(test_loader, netF, netB, netC, cfg)

            if acc_val >= best_acc:
                best_acc = acc_val
                best_netF = netF.state_dict()
                best_netB = netB.state_dict()
                best_netC = netC.state_dict()

            logging.info(f"[{epoch + 1}/{cfg.train.max_epoch}] {train_log} {val_log}")
        else:
            logging.info(f"[{epoch + 1}/{cfg.train.max_epoch}] {train_log}")

    torch.save(best_netF, "target_F.pt")
    torch.save(best_netB, "target_B.pt")
    torch.save(best_netC, "target_C.pt")

    logging.info(f"{cfg.dataset.name_src}=>{cfg.dataset.name_tar} Best Accuracy: {best_acc}")


if __name__ == '__main__':
    main()
