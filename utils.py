import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from torch import nn


class Scheduler:
    def __init__(self, max_iter, gamma=10, power=0.75):
        self.max_iter = max_iter
        self.gamma = gamma
        self.power = power

    def __call__(self, optimizer, iter_num):
        decay = (1 + self.gamma * iter_num / self.max_iter) ** (-self.power)
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr0'] * decay
            param_group['weight_decay'] = 1e-3
            param_group['momentum'] = 0.9
            param_group['nesterov'] = True
        return optimizer


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.count = 0
        self.sum = 0
        self.avg = 0
        self.val = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def topk_accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        wrong_k = batch_size - correct_k
        res.append(wrong_k.mul_(100.0 / batch_size))

    if len(res) == 1:
        return res[0]
    else:
        return res
    # return res, pred[:1].squeeze(0)


def Entropy(input_):
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy


def cal_acc(loader, netF, netB, netC, flag=False):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs = netC(netB(netF(inputs)))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(Entropy(all_output)).cpu().data.item()

    if flag:
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        acc = matrix.diagonal() / matrix.sum(axis=1) * 100
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = ' '.join(aa)
        return aacc, acc
    else:
        return accuracy * 100, mean_ent


def cal_acc_oda(loader, netF, netB, netC, epsilon, class_num):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs = netC(netB(netF(inputs)))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)
    ent = torch.sum(-all_output * torch.log(all_output + epsilon), dim=1) / np.log(class_num)
    ent = ent.float().cpu()
    initc = np.array([[0], [1]])
    kmeans = KMeans(n_clusters=2, random_state=0, init=initc, n_init=1).fit(ent.reshape(-1, 1))
    threshold = (kmeans.cluster_centers_).mean()

    predict[ent > threshold] = class_num
    matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
    matrix = matrix[np.unique(all_label).astype(int), :]

    acc = matrix.diagonal() / matrix.sum(axis=1) * 100
    unknown_acc = acc[-1:].item()

    return np.mean(acc[:-1]), np.mean(acc), unknown_acc
    # return np.mean(acc), np.mean(acc[:-1])


def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5  # unnormalize
    img = img.to('cpu')
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


def plot_classes_preds(preds, probs, images, labels, N_way, K_shot, K_query):
    """
    Plot Prediction Samples.
    Parameters
    ----------
    preds : list
        contain prediction value range from 0 to N_way - 1.
    probs : list
        contain prediction probability range from 0.0 ~ 1.0 formed softmax.
    images : list
        images[0] is sample images. Shape is (N_way, K_shot, 1, 28, 28).
        images[1] is query images. Shape is (N_way, K_shot, 1, 28, 28).
    labels: list
        labels[0] contains y value for sample image.
        labels[1] contains y value for query image.
    """
    # plot the images in the batch, along with predicted and true labels
    K_total = K_shot + K_query
    sample_images, query_images = images
    probs = [el[i].item() for i, el in zip(labels[1], probs)]

    fig = plt.figure(figsize=(N_way * 2, (K_total) * 3))

    # display sample images
    for row in np.arange(K_shot):
        for col in np.arange(N_way):
            ax = fig.add_subplot(K_total, N_way, row * N_way + col + 1, xticks=[], yticks=[])
            matplotlib_imshow(sample_images[col * K_shot + row], one_channel=True)
            ax.set_title(labels[0][col * K_shot + row].item())

    # display query images
    for row in np.arange(K_query):
        for col in np.arange(N_way):
            ax = fig.add_subplot(K_total, N_way, K_shot * N_way + row * N_way + col + 1, xticks=[],
                                 yticks=[])
            matplotlib_imshow(query_images[col * K_query + row], one_channel=True)
            ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
                preds[col * K_query + row],
                probs[col * K_query + row] * 100.0,
                labels[1][col * K_query + row]),
                color=(
                    "green" if preds[col * K_query + row] == labels[1][col * K_query + row].item() else "red"))

    fig.tight_layout()

    return fig
