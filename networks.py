from torch import nn
from torch.nn import functional as F


class Actor(nn.Module):

    def __init__(self, size):
        super(Actor, self).__init__()
        # convolution layers
        self.conv1 = nn.Conv2d(in_channels=12, out_channels=48, kernel_size=(5, 5), padding=2)
        self.batch_norm1 = nn.BatchNorm2d(48)
        self.conv2 = nn.Conv2d(in_channels=48, out_channels=48, kernel_size=(3, 3), padding=1)
        self.batch_norm2 = nn.BatchNorm2d(48)
        self.conv3 = nn.Conv2d(in_channels=48, out_channels=48, kernel_size=(3, 3), padding=1)
        self.batch_norm3 = nn.BatchNorm2d(48)
        self.conv4 = nn.Conv2d(in_channels=48, out_channels=48, kernel_size=(3, 3), padding=1)
        self.batch_norm4 = nn.BatchNorm2d(48)
        self.conv5 = nn.Conv2d(in_channels=48, out_channels=48, kernel_size=(3, 3), padding=1)
        self.batch_norm5 = nn.BatchNorm2d(48)
        self.conv6 = nn.Conv2d(in_channels=48, out_channels=48, kernel_size=(3, 3), padding=1)
        self.batch_norm6 = nn.BatchNorm2d(48)

        # dense softmax
        self.dense_in = 48 * size ** 2
        self.dense_out = 4 * ((size - 2) ** 2 + 3 * (size - 2) + 2) * 2
        self.dense = nn.Linear(in_features=self.dense_in, out_features=self.dense_out)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = self.batch_norm3(x)
        x = F.relu(x)

        x = self.conv4(x)
        x = self.batch_norm4(x)
        x = F.relu(x)

        x = self.conv5(x)
        x = self.batch_norm5(x)
        x = F.relu(x)

        x = self.conv6(x)
        x = self.batch_norm6(x)
        x = F.relu(x)

        x = x.view(-1, self.dense_in)

        x = self.dense(x)
        x = F.softmax(x)
        return x