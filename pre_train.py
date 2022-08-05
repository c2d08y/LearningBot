import os
import time

import torch.optim
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from networks import Actor
from dataset import FrameDataset
from const import *


def train(pai: Actor, dataset: FrameDataset, map_size, device, bs=1):
    """
    model   dataset     map_size
    pai20   non_maze    20
    pai10   non_maze1v1 10
    pai19   maze        19
    pai9    maze1v1     9
    :param pai: model
    :param dataset: dataset
    :param map_size: map_size
    :param device: device
    :param bs: batch_size
    :return: 
    """
    # data
    dataloader = DataLoader(dataset=dataset, batch_size=bs, shuffle=True)

    # loss function
    loss_fn = nn.CrossEntropyLoss().to(device)

    # optimizer
    learning_rate = 0.01
    optimizer = torch.optim.SGD(params=pai.parameters(), lr=learning_rate)

    # others
    epoch = 1                                  # train times
    trained_steps = 0                             # count steps already trained
    writer = SummaryWriter("pre_train_logs")    # graph drawer

    # train
    for i in range(epoch):
        print("-" * 20 + f"train epoch {i + 1}" + "-" * 20)

        for data in dataloader:
            # get data
            state = data["state"]
            target = data["action"]
            state = state.to(device)

            # from [x1, y1, x2, y2, is_half] to an index
            translator = ActionTranslator()
            target_i = torch.zeros([1, 2 * 4 * ((map_size - 2) ** 2 + 3 * (map_size - 2) + 2)]).to(device)
            t_i = translator.a_to_i(map_size, target)
            target_i[0][t_i] = 1

            # calculate loss
            output = pai(state)
            loss = loss_fn(output, target_i)

            # optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # output loss
            trained_steps += 1
            if trained_steps % 50 == 0:
                print(f"step={trained_steps}, loss={loss}")
                writer.add_scalar("loss", loss.item(), trained_steps)

    writer.close()


def get_model(model_path, map_size):
    if os.path.exists(model_path):
        return torch.load(model_path)
    else:
        return Actor(map_size)


def main():
    # check_gpu
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"train on device: {torch.cuda.get_device_name()}")
    else:
        device = torch.device("cpu")
        print("train on device: cpu")

    pai20 = get_model("./model/non_maze.pth", 20).to(device)
    # pai10 = get_model("./model/non_maze1v1.pth", 10).to(device)
    # pai19 = get_model("./model/maze.pth", 19).to(device)
    # pai9 = get_model("./model/maze1v1.pth", 9).to(device)

    train(pai20, FrameDataset("./Datasets/non_maze/"), 20, device)
    # train(pai10, FrameDataset("./Datasets/non_maze1v1/"), 10, device)
    # train(pai19, FrameDataset("./Datasets/maze/"), 19, device)
    # train(pai9, FrameDataset("./Datasets/maze1v1/"), 9, device)

    torch.save(pai20, "./model/non_maze.pth")
    # torch.save(pai10, "./model/non_maze1v1.pth")
    # torch.save(pai19, "./model/maze.pth")
    # torch.save(pai9, "./model/maze1v1.pth")


if __name__ == '__main__':
    main()