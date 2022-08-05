import os
import pickle
from torch.utils.data import Dataset


class FrameDataset(Dataset):

    def __init__(self, dataset_path="./Datasets/"):
        super(FrameDataset, self).__init__()
        self.path = dataset_path

    def __len__(self):
        return len(os.listdir(self.path + "/state/"))

    def __getitem__(self, item):
        s_f = open(self.path + "/state/" + str(item + 1), "rb")
        state = pickle.load(s_f)
        s_f.close()
        t_f = open(self.path + "/target/" + str(item + 1), "rb")
        action = pickle.load(t_f)
        t_f.close()
        return {"state": state, "action": action}