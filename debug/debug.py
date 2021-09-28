from __future__ import print_function

import os.path
import pickle
import random
import sys
sys.path.append('../')


import numpy as np
import torch
import torch.nn as nn
from models.util import *
# from util import *

from transform.transform_util import *
from dataset.cifartask import *
from label_prop.label_prop import *


def main():
    # parser = argparse.ArgumentParser('argument for training')
    #
    # parser.add_argument('--eval_freq', type=int, default=10, help='meta-eval frequency')
    # parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    # parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    # parser.add_argument('--save_freq', type=int, default=10, help='save frequency')
    # parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    # parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    # parser.add_argument('--epochs', type=int, default=100, help='number of training epochs')
    #
    # # optimization
    # parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    # parser.add_argument('--lr_decay_epochs', type=str, default='60,80', help='where to decay lr, can be a list')
    # parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    # parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    # parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    # parser.add_argument('--adam', action='store_true', help='use adam optimizer')
    #
    # # dataset
    # parser.add_argument('--model', type=str, default='resnet12', choices=model_pool)
    # parser.add_argument('--dataset', type=str, default='miniImageNet', choices=['miniImageNet', 'tieredImageNet',
    #                                                                             'CIFAR-FS', 'FC100'])
    # parser.add_argument('--transform', type=str, default='A', choices=transforms_list)
    # parser.add_argument('--use_trainval', action='store_true', help='use trainval set')
    #
    # # cosine annealing
    # parser.add_argument('--cosine', action='store_true', help='using cosine annealing')
    #
    # # specify folder
    # parser.add_argument('--model_path', type=str, default='', help='path to save model')
    # parser.add_argument('--tb_path', type=str, default='', help='path to tensorboard')
    # parser.add_argument('--data_root', type=str, default='', help='path to data root')
    #
    # # meta setting
    # parser.add_argument('--n_test_runs', type=int, default=600, metavar='N',
    #                     help='Number of test runs')
    # parser.add_argument('--n_ways', type=int, default=5, metavar='N',
    #                     help='Number of classes for doing each classification run')
    # parser.add_argument('--n_shots', type=int, default=1, metavar='N',
    #                     help='Number of shots in test')
    # parser.add_argument('--n_queries', type=int, default=15, metavar='N',
    #                     help='Number of query in test')
    # parser.add_argument('--n_aug_support_samples', default=5, type=int,
    #                     help='The number of augmented samples for each meta test sample')
    # parser.add_argument('--test_batch_size', type=int, default=1, metavar='test_batch_size',
    #                     help='Size of test batch)')
    #
    # parser.add_argument('-t', '--trial', type=str, default='1', help='the experiment id')
    #
    # opt = parser.parse_args()
    #
    # if opt.dataset == 'CIFAR-FS' or opt.dataset == 'FC100':
    #     opt.transform = 'D'
    #
    # if opt.use_trainval:
    #     opt.trial = opt.trial + '_trainval'
    #
    # # set the path according to the environment
    # if not opt.model_path:
    #     opt.model_path = './models_pretrained'
    # if not opt.tb_path:
    #     opt.tb_path = './tensorboard'
    # if not opt.data_root:
    #     opt.data_root = './data/{}'.format(opt.dataset)
    # else:
    #     opt.data_root = '{}/{}'.format(opt.data_root, opt.dataset)
    # opt.data_aug = True
    #
    # iterations = opt.lr_decay_epochs.split(',')
    # opt.lr_decay_epochs = list([])
    # for it in iterations:
    #     opt.lr_decay_epochs.append(int(it))
    #
    # opt.model_name = '{}_{}_lr_{}_decay_{}_trans_{}'.format(opt.model, opt.dataset, opt.learning_rate,
    #                                                         opt.weight_decay, opt.transform)
    #
    # if opt.cosine:
    #     opt.model_name = '{}_cosine'.format(opt.model_name)
    #
    # if opt.adam:
    #     opt.model_name = '{}_useAdam'.format(opt.model_name)
    #
    # opt.model_name = '{}_trial_{}'.format(opt.model_name, opt.trial)
    #
    # opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    # if not os.path.isdir(opt.tb_folder):
    #     os.makedirs(opt.tb_folder)
    #
    # opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    # if not os.path.isdir(opt.save_folder):
    #     os.makedirs(opt.save_folder)
    #
    # opt.n_gpu = torch.cuda.device_count()
    # path = '../save/model/teacher/mini_distilled.pth'
    #
    # model = models.resnet12(num_classes=64)
    #
    # model.load_state_dict(torch.load(path)['model'])
    #
    # print(model)

    # print(dict(dict(torch.load(path))['model']).keys())

    # data_root = '../save/data/CIFAR-FS/CIFAR-FS'
    #
    # file_pattern = '%s.pickle'
    #
    # pattern = 'train'
    #
    # with open(os.path.join(data_root, file_pattern % pattern), 'rb') as f:
    #     data = pickle.load(f, encoding='latin1')
    #     imgs = data['data']
    #     labels = data['labels']

    # class Model(nn.Module):
    #     def __init__(self):
    #         super(Model, self).__init__()
    #         self.w = nn.Parameter(torch.randn(3, 3, requires_grad=True))
    #         self.b = nn.Parameter(torch.randn(3, requires_grad=True))
    #         self.register_parameter('weight', self.w)
    #         self.register_parameter('beta', self.b)
    #
    #     def forward(self, x):
    #         return torch.matmul(x, self.w) + self.b
    #
    #
    # model = Model()
    # model.train()
    #
    # criterion = ContrastiveLoss(0.05, 0.5)
    #
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    # for epoch in range(10):
    #     x1 = model(inputs)
    #     x2 = model(torch.exp(inputs))
    #     loss = criterion(x1, x2)
    #     print('epoch:{} loss:{}'.format(epoch, loss))
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()

    # opt.data_root = '../save/data/CIFAR-FS/CIFAR-FS'
    # path = '../save/model/ckpt/S:resnet12_T:resnet12_CIFAR-FS_kd_r:0.5_a:0.5_b:0_trans_D_born3/resnet12_last.pth'
    # model = create_model('resnet12', 64, 'CIFAR-FS')
    # model.load_state_dict(torch.load(path)['model'])
    # model.cuda()
    #
    # _, test_trans = transforms_options['D']
    # test_loader = DataLoader(CIFAR100(opt, partition='train', transform=test_trans), batch_size=opt.batch_size // 2,
    #                          shuffle=True)
    #
    # criterion = nn.CrossEntropyLoss()
    #
    # validate(test_loader, model, criterion, opt)

    data_root = '../save/data/CIFAR-FS/CIFAR-FS'
    ckpt = 34
    path = '../save/model/ckpt/S:resnet12_T:resnet12_CIFAR-FS_kd_r:0.5_a:0.5_b:0_trans_D_born3/resnet12_last.pth'
    num_class = 64
    print('============ loading model... ============')
    model = create_model('resnet12', num_class, 'CIFAR-FS')
    model.load_state_dict(torch.load(path)['model'])

    # print('============ preparing data... ============')
    # train_trans, test_trans = transforms_options['D']
    #
    # train_loader = DataLoader(CIFAR100(opt, transform=train_trans), batch_size=1, shuffle=True)
    #
    # total_count = 100
    # print('============ total count: {} ============'.format(total_count))
    # features = []
    # labels = []
    # print('============ calculating features... ============')
    # for idx, (input, target, _) in enumerate(train_loader):
    #     if idx == total_count: break
    #     input = input.float()
    #     if torch.cuda.is_available():
    #         input = input.cuda()
    #         target = target.cuda()
    #     labels.append(target.cpu().detach().item())
    #
    #     feat = model(input, True)[0][-1].cpu()
    #     print(feat.shape)
    #     features.append(feat.detach().numpy())
    # print('============ done ============')
    # features = np.asarray(features)
    # label2label = {}
    # cur_class = 0
    # for _, label in enumerate(labels):
    #     if label not in label2label:
    #         label2label[label] = cur_class
    #         cur_class += 1
    # new_labels = []
    # for _, label in enumerate(labels):
    #     new_labels.append(label2label[label])
    #
    # new_labels = np.asarray(new_labels)
    # print('=========== new labels: ===============')
    # print(new_labels)
    # print('=======================================')
    #
    # num_labeled = (int)(0.2 * total_count)
    # num_unlabeled = total_count - num_labeled
    #
    # mat_labeled = features[0:num_labeled]
    # mat_unlabeled = features[num_labeled:total_count]
    # label_set = new_labels[0:num_labeled]
    #
    # print('=========== propagation... ===============')
    # predict = labelPropagation(mat_labeled, mat_unlabeled, label_set, max_iter=5000)
    #
    # print('=========== comparing: ===============')
    # print(predict)
    # print(new_labels[num_labeled:total_count])

    task = CIFARTask(data_root)

    save_path = '../save/test'

    fine_tune_with_unlabeled(save_path,task,model)



if __name__ == '__main__':
    main()
