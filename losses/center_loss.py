import torch


def compute_center_loss(features, centers, targets):
    features = features.reshape(features.size(0), -1)
    target_centers = centers[targets]
    criterion = torch.nn.MSELoss()
    center_loss = criterion(features, target_centers)
    return center_loss


def get_center_delta(features, centers, targets, alpha):
    features = features.data
    features = features.reshape(features.size(0), -1)
    targets, indices = torch.sort(targets)
    target_centers = centers[targets]
    features = features[indices]

    delta_centers = target_centers - features
    uni_targets, indices = torch.unique(
        targets.cpu(), sorted=True, return_inverse=True)

    uni_targets = uni_targets.cuda()
    indices = indices.cuda()

    delta_centers = torch.zeros(
        uni_targets.size(0), delta_centers.size(1)
    ).cuda().index_add_(0, indices, delta_centers)

    targets_repeat_num = uni_targets.size()[0]
    uni_targets_repeat_num = targets.size()[0]
    targets_repeat = targets.repeat(
        targets_repeat_num).reshape(targets_repeat_num, -1)
    uni_targets_repeat = uni_targets.unsqueeze(1).repeat(
        1, uni_targets_repeat_num)
    same_class_feature_count = torch.sum(
        targets_repeat == uni_targets_repeat, dim=1).float().unsqueeze(1)

    delta_centers = delta_centers / (same_class_feature_count + 1.0) * alpha
    result = torch.zeros_like(centers)
    result[uni_targets, :] = delta_centers
    return result
