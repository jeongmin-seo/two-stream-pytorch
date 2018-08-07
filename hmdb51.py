import torch
import os
import re
from torch.utils import data

class Dataset(data.Dataset):

    def __init__(self, load_type, txt_path, transforms=None):
        self._load_type = load_type
        self._txt_path = txt_path
        if load_type == 'train':
            self._train_data_list = []
        elif load_type == 'test':
            self._test_data = {}
        self.set_data_list()

    def __len__(self):
        pass

    def __getitem__(self, item):

        if self._load_type == 'train':
            pass
        elif self._load_type == 'test':


        X = "data"
        y = "label"

        return X, y


