import torch
import torchvision.transforms as transforms
import random
import torch.nn as nn
import torch.nn.functional as F
from dataset.cifartask import CIFARTask
from torch.utils.data import SequentialSampler, BatchSampler
from util import AverageMeter
from label_prop.label_prop import *
import os

def sample_transform():
    choices = [transforms.RandomCrop(32, padding=4),
               transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
               transforms.RandomHorizontalFlip()]

    trans = random.sample(choices, 2)
    return trans


class ContrastiveLoss(nn.Module):
    def __init__(self, T=0.15, alpha=1):
        super(ContrastiveLoss, self).__init__()
        self.T = T
        self.alpha = alpha

    def forward(self, x, y):
        exp_pos = []
        exp_neg = []

        num_samples = len(x)
        for i in range(num_samples):
            exp_pos.append(self.get_exp_cosine(x[i], y[i], self.T))

        sum = torch.zeros(1).cuda()
        for i in range(num_samples):
            for j in range(num_samples):
                if i != j:
                    sum = torch.add(sum, self.get_exp_cosine(x[i], x[j], self.T))
                    sum = torch.add(sum, self.get_exp_cosine(x[i], y[j], self.T))
            exp_neg.append(sum)

        log_items = [torch.log(torch.div(exp_pos[i], exp_neg[i])) for i in range(num_samples)]
        nce_loss = torch.zeros_like(log_items[0])
        for item in log_items:
            nce_loss = torch.add(nce_loss, item)
        nce_loss = -nce_loss
        js_loss = self.js_div(x, y)
        loss = nce_loss + self.alpha * js_loss
        return loss

    def get_exp_cosine(self, a, b, T):
        return torch.exp(torch.cosine_similarity(a, b, dim=(int)(len(a.size()) > 1)) / T)

    def js_div(self, a: torch.Tensor, b, is_softmax=True):
        kl_div = nn.KLDivLoss(reduction='batchmean')
        if is_softmax:
            a = F.softmax(a, dim=(int)(len(a.size()) > 1))
            b = F.softmax(b, dim=(int)(len(b.size()) > 1))
        log_mean = torch.log((a + b) / 2)

        return (kl_div(log_mean, a) + kl_div(log_mean, b)) / 2


def fine_tune_with_unlabeled(save_path, task: CIFARTask, model: nn.Module, batch_size=80, epochs=100,begin_epoch=1):
    model.cuda()
    # sample 2 transforms for contrastive learning
    trans_x, trans_y = sample_transform()
    trans_x = task.task_trans(trans_x)
    trans_y = task.task_trans(trans_y)

    # get a copy of unlabeled sample pool and shuffle it
    unlabeled_samples = list(task.get_raw_sample('u'))
    random.shuffle(unlabeled_samples)

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    criterion = ContrastiveLoss()

    # build batches
    batches = []
    begin = 0
    for _ in range(task.total_unlabeled // batch_size + 1):
        end = min(begin + batch_size, task.sample_num)
        batches.append(unlabeled_samples[begin:end])
        begin = end

    for epoch in range(begin_epoch, epochs + 1):
        cosine_scheduler.step()

        fine_tune_one_epoch(epoch, model, batches, criterion, optimizer, (trans_x, trans_y))
        # test epoch acc
        model.eval()
        acc = predict(task,model)

        print('epoch [{}] Acc [{}]'.format(epoch,acc))

        print('==> Saving....')
        state = {
            'epoch': epoch,
            'model': model.state_dict()
        }
        save_file = os.path.join(save_path, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
        torch.save(state, save_file)




def fine_tune_one_epoch(epoch, model, batches, criterion, optimizer, trans_list):
    model.train()
    losses = AverageMeter()
    trans_x, trans_y = trans_list

    for idx, batch in enumerate(batches):
        samples_x = []
        samples_y = []
        for sample in batch:
            samples_x.append(trans_x(sample))
            samples_y.append(trans_y(sample))
        samples_x = torch.stack(samples_x).float()
        samples_y = torch.stack(samples_y).float()

        if torch.cuda.is_available():
            samples_x = samples_x.cuda()
            samples_y = samples_y.cuda()

        feat_x = model(samples_x, is_feat=True)[0][-1]
        feat_y = model(samples_y, is_feat=True)[0][-1]

        loss = criterion(feat_x, feat_y)

        # =================optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # ===================
        losses.update(loss.item(), len(batch))
        print('epoch [{}]: idx [{}/{}  loss [{}]/[{}]'.format(epoch, idx+1, len(batches), losses.val, losses.avg))
