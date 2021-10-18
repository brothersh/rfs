import torch.nn as nn





class MLPEmbedding(nn.Module):
    def __init__(self, in_channel, n_cls, out_channel=128):
        super(MLPEmbedding, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channel, in_channel // 2),
            nn.ReLU(inplace=True),
            nn.Linear(in_channel // 2, out_channel),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Linear(out_channel, n_cls)

    def forward(self, x):
        x = self.mlp(x)
        feat = x
        x = self.classifier(x)
        return feat, x
