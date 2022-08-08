import torch
import numpy as np


class ReplayBuffer(object):

    def __init__(self, args: dict):
        self.s = np.zeros((args["batch_size"], args["map_size"]))
        self.a = np.zeros((args["batch_size"], 1))
        self.a_log_prob = np.zeros((args["batch_size"], 1))
        self.r = np.zeros((args["batch_size"], 1))
        self.s_ = np.zeros((args["batch_size"], args["state_dim"]))
        self.done = np.zeros((args["batch_size"], 1))
        self.size = 0

    def __len__(self):
        return self.size

    def clear(self):
        self.size = 0

    def store(self, s, a, a_log_prob, r, s_, done):
        self.s[self.size] = s
        self.a[self.size] = a
        self.a_log_prob[self.size] = a_log_prob
        self.r[self.size] = r
        self.s_[self.size] = s_
        self.done[self.size] = done
        self.size += 1

    def to_tensor(self):
        s = torch.tensor(self.s, dtype=torch.float)
        a = torch.tensor(self.a, dtype=torch.long)
        a_log_prob = torch.tensor(self.a_log_prob, dtype=torch.float)
        r = torch.tensor(self.r, dtype=torch.float)
        s_ = torch.tensor(self.s_, dtype=torch.float)
        done = torch.tensor(self.done, dtype=torch.float)

        return s, a, a_log_prob, r, s_, done