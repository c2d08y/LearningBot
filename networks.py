from torch import nn


def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)


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

        # dense & softmax
        self.dense_in = 48 * size ** 2
        self.dense_out = 4 * ((size - 2) ** 2 + 3 * (size - 2) + 2) * 2
        self.dense1 = nn.Linear(in_features=self.dense_in, out_features=self.dense_out)
        self.dense2 = nn.Linear(in_features=self.dense_out, out_features=self.dense_out)
        self.softmax = nn.Softmax(dim=0)

        # activation function
        self.activ_func = nn.Tanh()

        # init
        orthogonal_init(self.conv1)
        orthogonal_init(self.conv2)
        orthogonal_init(self.conv3)
        orthogonal_init(self.conv4)
        orthogonal_init(self.dense1)
        orthogonal_init(self.dense2, gain=0.01)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.activ_func(x)

        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.activ_func(x)

        x = self.conv3(x)
        x = self.batch_norm3(x)
        x = self.activ_func(x)

        x = self.conv4(x)
        x = self.batch_norm4(x)
        x = self.activ_func(x)

        x = x.view(-1, self.dense_in)

        x = self.dense1(x)
        x = self.activ_func(x)

        x = self.dense2(x)
        x = self.softmax(x)
        return x


class Critic(nn.Module):

    def __init__(self, size):
        super(Critic, self).__init__()
        # convolution layers
        self.conv1 = nn.Conv2d(in_channels=12, out_channels=48, kernel_size=(5, 5), padding=2)
        self.batch_norm1 = nn.BatchNorm2d(48)
        self.conv2 = nn.Conv2d(in_channels=48, out_channels=48, kernel_size=(3, 3), padding=1)
        self.batch_norm2 = nn.BatchNorm2d(48)
        self.conv3 = nn.Conv2d(in_channels=48, out_channels=48, kernel_size=(3, 3), padding=1)
        self.batch_norm3 = nn.BatchNorm2d(48)
        self.conv4 = nn.Conv2d(in_channels=48, out_channels=48, kernel_size=(3, 3), padding=1)
        self.batch_norm4 = nn.BatchNorm2d(48)

        # dense
        self.dense_in = 48 * size ** 2
        self.dense_out = 4 * ((size - 2) ** 2 + 3 * (size - 2) + 2) * 2
        self.dense1 = nn.Linear(in_features=self.dense_in, out_features=self.dense_out)
        self.dense2 = nn.Linear(in_features=self.dense_out, out_features=1)

        # activation function
        self.activ_func = nn.Tanh()

        # init
        orthogonal_init(self.conv1)
        orthogonal_init(self.conv2)
        orthogonal_init(self.conv3)
        orthogonal_init(self.conv4)
        orthogonal_init(self.dense1)
        orthogonal_init(self.dense2, gain=0.01)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.activ_func(x)

        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.activ_func(x)

        x = self.conv3(x)
        x = self.batch_norm3(x)
        x = self.activ_func(x)

        x = self.conv4(x)
        x = self.batch_norm4(x)
        x = self.activ_func(x)

        x = x.view(-1, self.dense_in)

        x = self.dense1(x)
        x = self.activ_func(x)

        x = self.dense2(x)
        return x