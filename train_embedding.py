import os
from pathlib import Path
import torch.cuda
import torch.nn as nn
import torch.nn.functional as F

from models.embedding import MLPEmbedding
from debug_util import get_timestamp


def train_mlp(mlp: MLPEmbedding, backbone, loader, optimizer, max_epoch, save_path, logger):
    mlp.train()
    backbone.eval()
    criterion_cls = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        mlp.cuda()
        backbone.cuda()
        criterion_cls.cuda()
    for epoch in range(1, max_epoch + 1):
        # ==========forward==================
        for idx, (inputs, labels, _, _) in enumerate(loader):
            inputs = inputs.float().cuda()
            labels = labels.cuda()
            with torch.no_grad():
                feats = backbone(inputs, is_feat=True)[0][-1]
            _, logit = mlp(feats)
            loss_cls = criterion_cls(logit, labels)
            loss = loss_cls
            logger.log_value('train_loss', loss, epoch)

            # ===========backward=================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % 10 == 0:
            print('epoch[{}]|[{}]  loss[{}]'.format(epoch, max_epoch, loss))

    state = {
        'model': mlp.state_dict() if torch.cuda.device_count() <= 1 else mlp.module.state_dict(),
    }

    file = Path(save_path)
    if not file.exists() or not file.is_dir():
        file.mkdir()
    save_file = os.path.join(save_path, 'mlp_last_%s.pth' % get_timestamp())
    torch.save(state, save_file)
