import torch


class ReplayBuffer(object):

    def __init__(self, args: dict):
        self.s = torch.zeros((args["batch_size"], 12, args["state_dim"], args["state_dim"]))
        self.a = torch.zeros((args["batch_size"], 1))
        self.a_log_prob = torch.zeros((args["batch_size"], 1))
        self.r = torch.zeros((args["batch_size"], 1))
        self.s_ = torch.zeros((args["batch_size"], 12, args["state_dim"], args["state_dim"]))
        self.done = torch.zeros((args["batch_size"], 1))
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

    def get_data(self):
        return self.s, self.a, self.a_log_prob, self.r, self.s_, self.done