from __future__ import print_function

import os
import pickle
from PIL import Image
import numpy as np
import random

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class MyCIFAR100(Dataset):
    """support FC100 and CIFAR-FS"""

    def __init__(self, data_root, partition='train', pretrain=True, is_sample=False, k=4096,
                 transform=None, label_map=None):
        super(Dataset, self).__init__()
        self.data_root = data_root
        self.partition = partition
        self.mean = [0.5071, 0.4867, 0.4408]
        self.std = [0.2675, 0.2565, 0.2761]
        self.normalize = transforms.Normalize(mean=self.mean, std=self.std)
        self.pretrain = pretrain

        if transform is None:
            if self.partition == 'train':
                self.transform = transforms.Compose([
                    lambda x: Image.fromarray(x),
                    transforms.RandomCrop(32, padding=4),
                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                    transforms.RandomHorizontalFlip(),
                    lambda x: np.asarray(x),
                    transforms.ToTensor(),
                    self.normalize
                ])
            else:
                self.transform = transforms.Compose([
                    lambda x: Image.fromarray(x),
                    transforms.ToTensor(),
                    self.normalize
                ])
        else:
            self.transform = transform

        self.file_pattern = 'new%s.pickle'

        with open(os.path.join(self.data_root, self.file_pattern % self.partition), 'rb') as f:
            data = pickle.load(f, encoding='latin1')
            self.imgs = data['data']
            labels = data['labels']
            self.real2fake = label_map
            if self.real2fake is None:
                self.real2fake = dict()
                cur_class = 0
                for idx, label in enumerate(labels):
                    if label not in self.real2fake:
                        self.real2fake[label] = cur_class
                        cur_class += 1
            new_labels = []
            for idx, label in enumerate(labels):
                new_labels.append(self.real2fake[label])
            self.labels = new_labels

        # contrastive sampling
        self.k = k
        self.is_sample = is_sample
        if self.is_sample:
            num_classes = len(set(self.labels))

            self.cls_positive = [[] for _ in range(num_classes)]
            for i in range(len(self.imgs)):
                self.cls_positive[self.labels[i]].append(i)

            self.cls_negative = [[] for _ in range(num_classes)]
            for i in range(len(self.imgs)):
                for j in range(len(self.imgs)):
                    if j != i:
                        self.cls_negative[i].extend(self.cls_positive[j])
            self.cls_positive = [np.asarray(self.cls_positive[i]) for i in range(num_classes)]
            self.cls_negative = [np.asarray(self.cls_negative[i]) for i in range(num_classes)]
            self.cls_positive = np.asarray(self.cls_positive)
            self.cls_negative = np.asarray(self.cls_negative)

    def __getitem__(self, item):
        img = np.asarray(self.imgs[item]).astype('uint8')
        img = self.transform(img)
        target = self.labels[item]

        if not self.is_sample:
            return img, target, item
        else:
            pos_idx = item
            replace = self.k > len(self.cls_negative[target])
            neg_idx = np.random.choice(self.cls_negative[target], self.k, replace=replace)
            sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx))
            return img, target, item, sample_idx

    def __len__(self):
        return len(self.labels)
