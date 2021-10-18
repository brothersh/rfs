from __future__ import print_function

import os
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torchvision.models

sys.path.append('../')

import torch
from torchsummary import summary
import torch.backends.cudnn as cudnn
import tensorboard_logger as tb_logger
from torch.utils.data import DataLoader
from torchvision import transforms
from models.util import *
from util import *

from train_embedding import train_mlp
from transform.transform_util import fine_tune_with_unlabeled
from dataset.mycifar import MyCIFAR100
from dataset.mymini import MyImageNet
from dataset.cifar import CIFAR100
from dataset.dataset_task import MiniTask, CIFARTask
from models.embedding import MLPEmbedding
from label_prop.label_prop import labelPropagation
from sklearn.manifold import TSNE
from debug_util import \
    get_trainval_feature, \
    print_scatter_3d, \
    print_scatter_2d, \
    build_new_train_val, \
    build_new_train_val_from_trainval, \
    print_task_feature_tsne, \
    get_timestamp


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'

    path = '../save/model/ckpt/'
    type = ['wrn_28_10', 'resnet12']
    dset = ['miniImageNet', 'CIFAR-FS']
    type_idx = 0
    dset_idx = 0
    model_name = '{}_{}_lr_0.05_decay_0.0005_trans_A_trial_newpretrain_with_center_loss'.format(type[type_idx],
                                                                                                dset[dset_idx])
    model_postfix = '{}_last.pth'.format(type[type_idx])
    model = create_model(type[type_idx], 80, dset[dset_idx])
    model.load_state_dict(torch.load(os.path.join(path, model_name, model_postfix))['model'])
    model.cuda()
    model.eval()
    data_root = '../save/data/miniImageNet/'

    save_path = '../save/embedding'
    finetune_model_save_path = '../save/model/test/{}'.format(model_name)
    #
    # 测试mlp
    # mlp_name = 'backbone:{}_with_center_loss|data:{}'.format(type[type_idx], dset[dset_idx])
    # log_dir = '../tensor_log/{}'.format(mlp_name)
    # logger = tb_logger.Logger(logdir=log_dir, flush_secs=2)
    #
    # task = MiniTask(data_root, item_range='s')
    # mlp = MLPEmbedding(in_channel=640, n_cls=task.n_way)
    # if torch.cuda.device_count() > 1:
    #     mlp = nn.DataParallel(mlp)
    # cudnn.benchmark = True
    # optimizer = torch.optim.SGD(mlp.parameters(), lr=0.05, weight_decay=0.0005, momentum=0.9)
    # loader = DataLoader(task, batch_size=task.total_support // 2, shuffle=True,num_workers=8)
    #
    # train_mlp(mlp, model, loader, optimizer, max_epoch=100, save_path=os.path.join(save_path, mlp_name), logger=logger)
    #
    # task.set_range('t')
    # loader = DataLoader(task, batch_size=16)
    # mlp.eval()
    # with torch.no_grad():
    #     embed = []
    #     lab = []
    #     for idx, (inputs, labels, targets, _) in enumerate(loader):
    #         inputs = inputs.float().cuda()
    #         features = model(inputs, is_feat=True)[0][-1]
    #         embeddings, _ = mlp(features)
    #         embed.append(embeddings)
    #         lab.append(labels)
    #     embed = torch.vstack(embed).data.cpu().numpy()
    #     lab = np.asarray(torch.hstack(lab))
    #
    # tsne = TSNE(n_components=2,init='pca',random_state=0)
    # result = tsne.fit_transform(embed)
    #
    # embed_visual_path = '../visual/embedding/{}'.format(mlp_name)
    # file = Path(embed_visual_path)
    # if not file.exists() or not file.is_dir():
    #     file.mkdir()
    # print_scatter_2d(result,lab,save_path=embed_visual_path,f_name='test_mlp_embedding',title='mlp embedding')



    # 测试微调的效果
    task = MiniTask(data_root)
    fine_tune_with_unlabeled(finetune_model_save_path, task, model,batch_size=8)

    # 测试训练集样本的聚类
    # result_save_path = '../visual/wrn_mini_trainval_3d_with_center_loss'
    # for i in range(5):
    #     task = MiniTask(data_root=data_root, file_pattern='mini_classification_%s.pickle', support_num=100,
    #                     unlabeled_num=100, query_num=100, pattern='train')
    #     print_task_feature_tsne(model, task, title='task {}'.format(i), result_save_path=result_save_path, n_dim=3)

    # feat_list = get_batch_trainval_feature(model, DataType=MyImageNet, data_root=data_root, batch_size=300)
    # n_dim = 3
    # for idx, (feat, target) in enumerate(feat_list):
    #     tsne = TSNE(n_components=n_dim, random_state=0, init='pca')
    #     result = tsne.fit_transform(feat)
    #     n_cls = len(np.unique(target))
    #     result_save_path = '../visual/seres12_trainval_feat_pca_with-center_loss'
    #     file = Path(result_save_path)
    #     if not file.exists() or not file.is_dir():
    #         file.mkdir()
    #     if n_dim > 2:
    #         print_scatter_3d(result, target,
    #                          save_path=result_save_path,
    #                          f_name=model_name + '_batch_{}'.format(idx),
    #                          title='batch_{} | {}_cls'.format(idx, n_cls))
    #     else:
    #         print_scatter_2d(result, target,
    #                          save_path=result_save_path,
    #                          f_name=model_name + '_batch_{}'.format(idx),
    #                          title='batch_{} | {}_cls'.format(idx, n_cls))

    # 采样task然后评估骨架的聚类能力
    # result_save_path = '../visual/wrn_mini_task_3d_with_center_loss'
    # for i in range(10):
    #     task = MiniTask(data_root)
    #     print_task_feature_tsne(model, task, title='task%s' % i, result_save_path=result_save_path, n_dim=2)

    # 查看pickle内容
    # tmp_pattern = 'miniImageNet_category_split_train_phase_%s.pickle'
    # tmp_pattern = 'miniImageNet_category_split_%s.pickle'
    # tmp_pattern = 'mini_classification_%s.pickle'
    # with open(os.path.join(data_root,tmp_pattern  % 'val'), 'rb') as f:
    #     data = pickle.load(f, encoding='latin1')
    #     labels = data['labels']
    #     label_set = set(data['labels'])
    #     print(data['data'].shape)
    #     print(label_set)
    #     tmp = {label:0 for label in label_set}
    #     for label in labels:
    #         tmp[label] += 1
    #
    #     for label in tmp:
    #         print('label: {},len: {}'.format(label,tmp[label]))

    # for i in range(10):
    #     img = data['data'][i]
    #     label = data['labels'][i]
    #     plt.figure()
    #     plt.imshow(img)
    #     plt.title(label)
    #     plt.savefig(fname='sample%d.png' % i)


if __name__ == '__main__':
    main()
