import os
import pickle
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import random
import torchvision.transforms as transforms


class DataTask(Dataset):
    def __init__(self, data_root, n_way=5, support_num=1, unlabeled_num=40, query_num=15, pattern='test',
                 item_range='t'):
        super(DataTask, self).__init__()
        # task param
        self.n_way = n_way
        self.support_num = support_num
        self.unlabeled_num = unlabeled_num
        self.query_num = query_num
        self.total_support = support_num * n_way
        self.total_unlabeled = unlabeled_num * n_way
        self.total_query = query_num * n_way
        self.sample_num = n_way * (support_num + unlabeled_num + query_num)
        self.item_range = item_range
        # file info
        self.data_root = data_root
        self.pattern = pattern
        self.normalize = None
        self.transform = None

    def __len__(self):
        return 0

    def __getitem__(self, item):
        return None

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

    def task_trans(self, trans):
        self.transform = transforms.Compose([
            lambda x: Image.fromarray(x),
            trans,
            lambda x: np.asarray(x),
            transforms.ToTensor(),
            self.normalize
        ])

    def build_task_sets(self):
        # build index set of s,u,q
        with open(os.path.join(self.data_root, self.file_pattern % self.pattern), 'rb') as f:
            data = pickle.load(f, encoding='latin1')
            imgs = data['data']
            task = self.build_task(data['labels'], self.n_way, self.unlabeled_num, self.query_num, self.support_num)
            self.label_set = task[0]
            self.support_set = task[1]
            self.unlabeled_set = task[2]
            self.query_set = task[3]
        # label_map
        self.label_map = {target: label for label, target in enumerate(self.label_set)}
        # real samples and targets
        self.samples = []
        self.targets = []
        self.labels = []
        for _set in [self.support_set, self.unlabeled_set, self.query_set]:
            for cls in _set:
                for img_idx in _set[cls]:
                    self.samples.append(imgs[img_idx])
                    self.targets.append(cls)
                    self.labels.append(self.label_map[cls])

        self.set_range(self.item_range)

    def print_task(self):
        print('{} way {} shot task: {}'.format(self.n_way, self.support_num, str(self.__class__)))
        print('total sample: {}'.format(self.sample_num))
        print('support num: {} | query num: {} | unlabeled num: {}'.format(self.total_support, self.total_query,
                                                                           self.total_query))
        print('class_set: {}'.format(self.label_set))
        print('support: {}'.format(self.support_set))
        print('query: {}'.format(self.query_set))
        print('unlabeled: {}'.format(self.unlabeled_set))

    def set_range(self, range):
        self.item_range = range

        if self.item_range == 's':
            self.range_samples = [np.asarray(img).astype('uint8') for img in self.samples[0:self.total_support]]
            self.range_targets = self.targets[0:self.total_support]
            self.range_labels = self.labels[0:self.total_support]
        elif self.item_range == 'q':
            self.range_samples = [np.asarray(img).astype('uint8') for img in self.samples[-self.total_query:]]
            self.range_targets = self.targets[-self.total_query:]
            self.range_labels = self.labels[-self.total_query:]
        elif self.item_range == 'u':
            self.range_samples = [np.asarray(img).astype('uint8') for img in
                                  self.samples[self.total_support:self.total_support + self.total_unlabeled]]
            self.range_targets = self.targets[self.total_support:self.total_support + self.total_unlabeled]
            self.range_labels = self.labels[self.total_support:self.total_support + self.total_unlabeled]
        else:
            self.range_samples = self.samples
            self.range_targets = self.targets
            self.range_labels = self.labels


class CIFARTask(DataTask):
    def __init__(self, data_root, file_pattern='%s.pickle', n_way=5, support_num=1, unlabeled_num=40, query_num=15,
                 pattern='test', item_range='t'):
        super(CIFARTask, self).__init__(data_root, n_way=n_way, support_num=support_num, unlabeled_num=unlabeled_num,
                                        query_num=query_num, pattern=pattern, item_range=item_range)
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
        self.file_pattern = '%s.pickle'
        self.build_task_sets()

    def __getitem__(self, index: int):
        return self.transform(self.range_samples[index]), self.range_labels[index], self.range_targets[index], index

    def __len__(self) -> int:
        return len(self.range_samples)


class MiniTask(DataTask):
    def __init__(self, data_root, file_pattern='miniImageNet_category_split_%s.pickle', n_way=5, support_num=1,
                 unlabeled_num=40, query_num=15, pattern='test', item_range='t'):
        super(MiniTask, self).__init__(data_root, n_way=n_way, support_num=support_num, unlabeled_num=unlabeled_num,
                                       query_num=query_num, pattern=pattern, item_range=item_range)
        # normalization
        self.mean = [120.39586422 / 255.0, 115.59361427 / 255.0, 104.54012653 / 255.0]
        self.std = [70.68188272 / 255.0, 68.27635443 / 255.0, 72.54505529 / 255.0]
        self.normalize = transforms.Normalize(mean=self.mean, std=self.std)
        # transform
        self.transform = transforms.Compose([
            lambda x: Image.fromarray(x),
            transforms.ToTensor(),
            self.normalize
        ])
        self.file_pattern = file_pattern
        self.build_task_sets()

    def __getitem__(self, index: int):
        return self.transform(self.range_samples[index]), self.range_labels[index], self.range_targets[index], index

    def __len__(self) -> int:
        return len(self.range_samples)
