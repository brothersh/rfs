import pickle
import numpy as np
import torch
import os
import random
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image


class CIFARTask(Dataset):
    def __init__(self, data_root, n_way=5, support_num=1, unlabeled_num=40, query_num=15, pattern='test'):
        super(Dataset, self).__init__()
        # task param
        self.n_way = n_way
        self.support_num = support_num
        self.unlabeled_num = unlabeled_num
        self.query_num = query_num
        self.total_support = support_num * n_way
        self.total_unlabeled = unlabeled_num * n_way
        self.total_query = query_num * n_way
        self.sample_num = n_way * (support_num + unlabeled_num + query_num)

        # normalization
        self.mean = [0.5071, 0.4867, 0.4408]
        self.std = [0.2675, 0.2565, 0.2761]
        self.normalize = transforms.Normalize(mean=self.mean, std=self.std)
        self.transform = transforms.Compose([
            lambda x: Image.fromarray(x),
            transforms.ToTensor(),
            self.normalize
        ])
        # file info
        self.data_root = data_root
        self.file_pattern = '%s.pickle'

        # build index set of s,u,q
        with open(os.path.join(data_root, self.file_pattern % pattern), 'rb') as f:
            data = pickle.load(f, encoding='latin1')
            imgs = data['data']
            task = self.build_task(data['labels'], n_way, unlabeled_num, query_num, support_num)
            self.label_set = task[0]
            self.support_set = task[1]
            self.unlabeled_set = task[2]
            self.query_set = task[3]

        # real samples and targets
        self.samples = []
        self.targets = []
        for _set in [self.support_set, self.unlabeled_set, self.query_set]:
            for cls in _set:
                for img_idx in _set[cls]:
                    self.samples.append(imgs[img_idx])
                    self.targets.append(cls)

    def __getitem__(self, index: int):
        img = np.asarray(self.samples[index]).astype('uint8')
        return self.transform(img)

    def __len__(self) -> int:
        return self.sample_num

    def build_task(self, labels, n_way=5, unlabeled_num=40, query_num=15, support_num=1):
        # calculate sample numbers per class and random sample {n_way} classes as {label_set}
        num_samples_per_class = (unlabeled_num + query_num + support_num)
        class_set = np.unique(labels)
        label_set = random.sample(class_set.tolist(), n_way)
        print(label_set)

        # build a dictionary for {label_c : [img_c_1 , img_c_2 ,..., img_c_n]}
        # where {img_c_i} is the ith-index of label_c in the whole dataset
        label_index = dict()
        for idx, label in enumerate(labels):
            if label not in label_index:
                label_index[label] = []
            label_index[label].append(idx)

        # build s,u,q sets according to {label_set}
        # formulated as {label_c:[idx_of_img_c_1 , ... ,idx_of_img_c_n}
        support_set = dict()
        unlabeled_set = dict()
        query_set = dict()

        # each sampled class we random sample {num_samples_per_class} samples for s,u,q sets
        # and then shuffle the sampled list to divide into [s | u | q]
        for label in label_set:
            samples_of_label = random.sample(label_index[label], num_samples_per_class)
            random.shuffle(samples_of_label)
            support_set[label] = samples_of_label[0:support_num]
            unlabeled_set[label] = samples_of_label[support_num:support_num + unlabeled_num]
            query_set[label] = samples_of_label[-query_num:]

        return label_set, support_set, unlabeled_set, query_set

    def get_raw_sample(self, flag):
        samples = []
        if flag == 's':
            for img in self.samples[0:self.total_support]:
                samples.append(np.asarray(img).astype('uint8'))
        elif flag == 'u':
            for img in self.samples[self.total_support:self.total_support + self.total_unlabeled]:
                samples.append(np.asarray(img).astype('uint8'))
        else:
            for img in self.samples[-self.total_query:]:
                samples.append(np.asarray(img).astype('uint8'))
        return samples

    def task_trans(self, trans):
        return transforms.Compose([
            lambda x: Image.fromarray(x),
            trans,
            lambda x: np.asarray(x),
            transforms.ToTensor(),
            self.normalize
        ])
