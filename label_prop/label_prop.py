import time
import numpy as np

import torch
import random
from dataset.dataset_task import CIFARTask, MiniTask, DataTask
from torch.utils.data.dataloader import DataLoader


def navie_knn(dataSet, query, k):
    numSamples = dataSet.shape[0]

    ## step 1: calculate Euclidean distance
    diff = np.tile(query, (numSamples, 1)) - dataSet
    squaredDiff = diff ** 2
    squaredDist = np.sum(squaredDiff, axis=1)  # sum is performed by row

    ## step 2: sort the distance
    sortedDistIndices = np.argsort(squaredDist)
    if k > len(sortedDistIndices):
        k = len(sortedDistIndices)

    return sortedDistIndices[0:k]


# build a big graph (normalized weight matrix)
def buildGraph(MatX, kernel_type, rbf_sigma=None, knn_num_neighbors=None):
    num_samples = MatX.shape[0]
    affinity_matrix = np.zeros((num_samples, num_samples), np.float32)
    if kernel_type == 'rbf':
        if rbf_sigma == None:
            raise ValueError('You should input a sigma of rbf kernel!')
        for i in range(num_samples):
            row_sum = 0.0
            for j in range(num_samples):
                diff = MatX[i, :] - MatX[j, :]
                affinity_matrix[i][j] = np.exp(np.sum(diff ** 2) / (-2.0 * rbf_sigma ** 2))
                row_sum += affinity_matrix[i][j]
            affinity_matrix[i][:] /= row_sum
    elif kernel_type == 'knn':
        if knn_num_neighbors == None:
            raise ValueError('You should input a k of knn kernel!')
        for i in range(num_samples):
            k_neighbors = navie_knn(MatX, MatX[i, :], knn_num_neighbors)
            affinity_matrix[i][k_neighbors] = 1.0 / knn_num_neighbors
    else:
        raise NameError('Not support kernel type! You can use knn or rbf!')

    return affinity_matrix


# label propagation
def labelPropagation(Mat_Label, Mat_Unlabel, labels, kernel_type='rbf', rbf_sigma=1.5, \
                     knn_num_neighbors=10, max_iter=500, tol=1e-3):
    # initialize
    num_label_samples = Mat_Label.shape[0]
    num_unlabel_samples = Mat_Unlabel.shape[0]
    num_samples = num_label_samples + num_unlabel_samples
    labels_list = np.unique(labels)
    num_classes = len(labels_list)

    MatX = np.vstack((Mat_Label, Mat_Unlabel))
    clamp_data_label = np.zeros((num_label_samples, num_classes), np.float32)
    for i in range(num_label_samples):
        clamp_data_label[i][labels[i]] = 1.0

    label_function = np.zeros((num_samples, num_classes), np.float32)
    label_function[0: num_label_samples] = clamp_data_label
    label_function[num_label_samples: num_samples] = -1

    # graph construction
    affinity_matrix = buildGraph(MatX, kernel_type, rbf_sigma, knn_num_neighbors)

    # start to propagation
    iter = 0;
    pre_label_function = np.zeros((num_samples, num_classes), np.float32)
    changed = np.abs(pre_label_function - label_function).sum()

    while iter < max_iter and changed > tol:
        pre_label_function = label_function
        iter += 1

        # propagation
        label_function = np.dot(affinity_matrix, label_function)

        # clamp
        label_function[0: num_label_samples] = clamp_data_label

        # check converge
        changed = np.abs(pre_label_function - label_function).sum()

    print('propagation done in {}'.format(iter))
    # get terminate label of unlabeled data
    unlabel_data_labels = np.zeros(num_unlabel_samples)
    for i in range(num_unlabel_samples):
        unlabel_data_labels[i] = np.argmax(label_function[i + num_label_samples])

    return unlabel_data_labels


# def get_feature_from(_set, features, targets, model, trans, imgs, new_label_map):
#     for cls in _set:
#         for img_index in _set[cls]:
#             img = imgs[img_index]
#             img = trans(np.asarray(img).astype('uint8'))
#             img = img.expand((1, -1, -1, -1))
#             feat = model(img, is_feat=True)[0][-1]
#             feat = feat.reshape(-1)
#             features.append(feat.detach().numpy())
#             targets.append(new_label_map[cls])


def predict(task: DataTask, model):
    # build the real feature list and their labels for label propagation
    model.eval()

    task_loader = DataLoader(task, batch_size=task.sample_num)
    with torch.no_grad():
        _, (inputs, labels, _, _) = next(enumerate(task_loader))
        inputs = inputs.float().cuda()
        features = model(inputs, is_feat=True)[0][-1]

    features = features.data.cpu().numpy()
    labels = np.asarray(labels)

    predict = labelPropagation(features[0:task.total_support], features[task.total_support:],
                               targets[0:task.total_support], max_iter=5000)

    # the last {query_num * n_way} of the {predict} list is the prediction of queryset
    query_predict = predict[-task.total_query:]
    expected = labels[-task.total_query:]

    acc = np.equal(expected, query_predict).sum() * 100 / len(query_predict)
    return acc
