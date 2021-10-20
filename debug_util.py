import time
import pickle
import os
import random
from pathlib import Path

import numpy as np

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from dataset.mycifar import MyCIFAR100


def save_plt_fig(plt, save_path, f_name):
    pattern = '{}_{}.png'
    plt.savefig(fname=os.path.join(save_path, pattern.format(f_name, get_timestamp())))


def print_scatter_3d(result, label, save_path, f_name, title):
    result = normalize(result)
    fig = plt.figure()
    plt.title(title)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(result[:, 0], result[:, 1], result[:, 2], c=label)
    ax.view_init(4, -72)
    save_plt_fig(plt, save_path, f_name)


def print_scatter_2d(result, label, save_path, f_name, title):
    result = normalize(result)
    fig = plt.figure()
    plt.title(title)
    plt.scatter(result[:, 0], result[:, 1], c=label)
    save_plt_fig(plt, save_path, f_name)


def print_text(result, label, save_path, f_name, title):
    result = normalize(result)
    plt.figure()
    plt.title(title)
    for i in range(result.shape[0]):
        plt.text(result[i, 0], result[i, 1], str(label[i]), color=plt.cm.Set1(label[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})
    save_plt_fig(plt, save_path, f_name)


def normalize(result):
    with torch.no_grad():
        tmp = torch.as_tensor(result)
        tmp = F.normalize(tmp)
        result = tmp.data.cpu().numpy()
    return result


def get_timestamp():
    return time.strftime('%Y%m%d_%H%M%S', time.localtime())


# 对trainval重新进行2/8划分
def build_new_train_val(data_root, alpha=0.8):
    pattern = 'miniImageNet_category_split_train_phase_%s.pickle'

    train_file = open(os.path.join(data_root, pattern % 'train'), 'rb')
    val_file = open(os.path.join(data_root, pattern % 'val'), 'rb')

    train_data = pickle.load(train_file, encoding='latin1')
    val_data = pickle.load(val_file, encoding='latin1')

    train_img = train_data['data']
    val_img = val_data['data']

    train_label = train_data['labels']
    val_label = val_data['labels']

    train_total_set = set(range(len(train_img)))
    val_total_set = set(range(len(val_img)))

    train_sample_idx = set(random.sample(train_total_set, int(alpha * len(train_img))))
    train_remain_idx = train_total_set - train_sample_idx
    val_sample_idx = set(random.sample(val_total_set, int(alpha * len(val_img))))
    val_remain_idx = val_total_set - val_sample_idx

    new_train_img = []
    new_train_label = []

    new_val_img = []
    new_val_label = []

    for idx in train_sample_idx:
        new_train_img.append(train_img[idx])
        new_train_label.append(train_label[idx])

    for idx in train_remain_idx:
        new_val_img.append(train_img[idx])
        new_val_label.append(train_label[idx])

    for idx in val_sample_idx:
        new_train_img.append(val_img[idx])
        new_train_label.append(val_label[idx])

    for idx in val_remain_idx:
        new_val_img.append(val_img[idx])
        new_val_label.append(val_label[idx])

    new_train_data = dict()
    new_train_data['data'] = new_train_img
    new_train_data['labels'] = new_train_label

    new_val_data = dict()
    new_val_data['data'] = new_val_img
    new_val_data['labels'] = new_val_label

    train_out = open(os.path.join(data_root, 'mini_classification_train.pickle'), 'wb')
    val_out = open(os.path.join(data_root, 'mini_classification_val.pickle'), 'wb')

    pickle.dump(new_train_data, train_out)
    pickle.dump(new_val_data, val_out)


def build_new_train_val_from_trainval(data_root, pickle_file, alpha=0.8):
    with open(os.path.join(data_root, pickle_file), 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    imgs = data['data']
    labels = data['labels']
    label_set = set(labels)
    cls_idx = {label: [] for label in labels}
    for idx, label in enumerate(labels):
        cls_idx[label].append(idx)

    train_dict = {label: [] for label in labels}
    val_dict = {label: [] for label in labels}

    for label in label_set:
        total_num = len(cls_idx[label])
        train_num = int(alpha * total_num)
        val_num = total_num - train_num
        train_set = set(random.sample(cls_idx[label], train_num))
        val_set = set(cls_idx[label]) - train_set
        train_dict[label].extend(list(train_set))
        val_dict[label].extend(list(val_set))

    new_train = {'data': [], 'labels': []}
    new_val = {'data': [], 'labels': []}

    for label in train_dict:
        for img in train_dict[label]:
            new_train['data'].append(imgs[img])
            new_train['labels'].append(label)
    for label in val_dict:
        for img in val_dict[label]:
            new_val['data'].append(imgs[img])
            new_val['labels'].append(label)

    new_train['data'] = np.asarray(new_train['data'])
    new_train['labels'] = np.asarray(new_train['labels'])
    new_val['data'] = np.asarray(new_val['data'])
    new_val['labels'] = np.asarray(new_val['labels'])
    with open(os.path.join(data_root, 'mini_classification_train.pickle'), 'wb') as f:
        pickle.dump(new_train, f)
    with open(os.path.join(data_root, 'mini_classification_val.pickle'), 'wb') as f:
        pickle.dump(new_val, f)


# 获取一个task和通过model后的feature和target
def get_task_feature_from_task(model, task):
    task_loader = DataLoader(task, batch_size=16, shuffle=True)
    outs = []
    targets = []
    with torch.no_grad():
        for _, (inputs, label, target, _) in enumerate(task_loader):
            inputs = inputs.float().cuda()
            feat = model(inputs, is_feat=True)[0][-1]
            outs.append(feat)
            targets.append(target)
        targets = torch.hstack(targets)
        features = torch.vstack(outs)

        features = features.data.cpu().numpy()
        targets = np.asarray(targets)
    return task, features, targets


def get_trainval_feature(model, DataType, data_root, partition='train', n_cls=5):
    loader = DataLoader(DataType(data_root=data_root, partition=partition), batch_size=64)
    selected = random.sample(range(80), n_cls)
    cls_sample = {selection: [] for selection in selected}
    with torch.no_grad():
        for idx, (inputs, targets, _) in enumerate(loader):
            inputs = inputs.float().cuda()
            targets = targets.cuda()
            features = model(inputs, is_feat=True)[0][-1].data.cpu().numpy()
            targets = targets.data.cpu().numpy()

            for idx, target in enumerate(targets):
                if target in selected:
                    cls_sample[target].append(features[idx])

        feats = []
        labels = []
        for target in cls_sample:
            feats.extend(cls_sample[target])
            labels.extend([target] * len(cls_sample[target]))
        labels = np.asarray(labels)
        feats = np.asarray(feats)
    return feats, labels


def print_task_feature_tsne(model, task, title, result_save_path='../visual/task', n_dim=2):
    _, features, targets = get_task_feature_from_task(model, task)
    tsne = TSNE(n_components=n_dim, init='pca', random_state=0)
    result = tsne.fit_transform(features)
    file = Path(result_save_path)
    if not file.exists() or not file.is_dir():
        file.mkdir()
    if n_dim > 2:
        print_scatter_3d(result, targets, result_save_path, f_name=title, title=title)
    else:
        print_scatter_2d(result, targets, result_save_path, f_name=title, title=title)


def create_pickle_from_csv(csv_path, csv_file_name, img_path):
    return
