import random
import os
from pathlib import Path

import torch.nn as nn
import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader

from label_prop.label_prop import predict_lp, predict_sm, labelPropagation
from util import *
from dataset.dataset_task import CIFARTask, DataTask, MiniTask
from torchvision import transforms
from debug_util import print_task_feature_tsne, get_timestamp


def sample_transform():
    choices = [transforms.RandomCrop(84, padding=4),
               transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
               transforms.RandomHorizontalFlip()]

    trans = random.sample(choices, 2)
    return trans


class ContrastiveLoss(nn.Module):
    def __init__(self, T=0.15, alpha=1):
        super(ContrastiveLoss, self).__init__()
        self.T = T

    def forward(self, x, y):
        n = x.shape[0]
        t_list = torch.vstack([x, y])
        aff = []
        for i in range(2 * n):
            aff_item = []
            for j in range(2 * n):
                aff_item.append(self.exp_cosine(t_list[i], t_list[j], self.T))
            aff.append(aff_item)
        loss = 0
        for i in range(2 * n):
            pos = aff[i][i + n] if i < n else aff[i][i - n]
            neg = sum(aff[i]) - aff[i][i] - pos
            loss += torch.log(pos / neg)
        return -loss

    def exp_cosine(self, a, b, T):
        return torch.exp(torch.cosine_similarity(a, b, dim=0) / T)

    def js_div(self, a: torch.Tensor, b, is_softmax=True):
        kl_div = nn.KLDivLoss(reduction='batchmean')
        if is_softmax:
            a = F.softmax(a, dim=(int)(len(a.size()) > 1))
            b = F.softmax(b, dim=(int)(len(b.size()) > 1))
        log_mean = torch.log((a + b) / 2)

        return (kl_div(log_mean, a) + kl_div(log_mean, b)) / 2


def fine_tune_with_unlabeled(save_path, task: DataTask, model: nn.Module, batch_size=80, epochs=100, begin_epoch=1):
    model.cuda()
    # sample 2 transforms for contrastive learning
    trans_x, trans_y = sample_transform()
    task.set_range('u')
    unlabeled_loader = DataLoader(task, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    criterion = ContrastiveLoss()

    visual_save_path = '../visual/transform_finetune'
    now = get_timestamp()
    f = Path(os.path.join(save_path, now))
    if not f.exists() or not f.is_dir():
        f.mkdir(parents=True)

    acc = predict_sm(task, model)
    print('init acc: [{}]%'.format(acc))
    print_task_feature_tsne(model, task, title='epoch_0', result_save_path=visual_save_path)
    for epoch in range(begin_epoch, epochs + 1):
        cosine_scheduler.step()

        fine_tune_one_epoch(epoch, model, task, unlabeled_loader, criterion, optimizer, (trans_x, trans_y))
        print_task_feature_tsne(model, task, title='epoch_{}'.format(epoch), result_save_path=visual_save_path)
        # test epoch acc
        # acc = predict_lp(task, model)
        acc = predict_sm(task, model)

        if epoch % 10 == 0:
            print('epoch [{}] Acc [{}]'.format(epoch, acc))

            print('==> Saving....')
            state = {
                'epoch': epoch,
                'model': model.state_dict()
            }
            save_file = os.path.join(save_path, now, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(state, save_file)


def fine_tune_one_epoch(epoch, model, task, loader, criterion, optimizer, trans_list):
    model.train()
    losses = AverageMeter()
    trans_x, trans_y = trans_list

    enum_x = enumerate(loader)
    enum_y = enumerate(loader)

    for batch_idx in range(len(loader)):
        task.set_trans(trans_x)
        idx, (inputs_x, _, _, _) = next(enum_x)
        task.set_trans(trans_y)
        idx, (inputs_y, _, _, _) = next(enum_y)
        # =================forward==================
        inputs_x = inputs_x.float().cuda()
        inputs_y = inputs_y.float().cuda()

        feat_x = model(inputs_x, is_feat=True)[0][-1]
        feat_y = model(inputs_y, is_feat=True)[0][-1]

        feat_x = F.normalize(feat_x)
        feat_y = F.normalize(feat_y)

        loss = criterion(feat_x, feat_y)
        # ==================backward=============
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), inputs_x.shape[0])
        print(
            'epoch [{}]: idx [{}/{}  loss [{}]/[{}]'.format(epoch, batch_idx + 1, len(loader), losses.val, losses.avg))
