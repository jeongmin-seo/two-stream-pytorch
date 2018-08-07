import torch
import numpy as np
import os
import re
from torch.utils.data import Dataset

class HMDB(Dataset):

    def __init__(self, _data_root, _txt_root, _load_type, _split_num, transform=None):

        self._root_dir = _data_root
        self._load_type = _load_type
        self._txt_path = os.path.join(_txt_root, self._load_type+'_split%d.txt' %_split_num)
        self._data_list = []
        self._transform = transform
        self.set_data_list()

    def __len__(self):
        return len(self._data_list)

    def __getitem__(self, _idx):

        data_name, label = self._data_list[_idx]
        split_data_name = re.split(r"[-]*", data_name)
        dir_name = split_data_name[0] + '-' + split_data_name[1]
        file_path = os.path.join(self._root_dir, dir_name, data_name)

        return self._transform(np.load(file_path)), label, dir_name

    @staticmethod
    def data_dir_reader(_txt_path):
        tmp = []
        f = open(_txt_path, 'r')
        for line in f.readlines():
            line = line.replace('\n', '')
            split_line = re.split(r"[\s+,\/]*", line)
            file_info = split_line[0] + "-%05d" % int(split_line[1])
            tmp.append([file_info, int(split_line[-1])])

        return tmp

    def set_data_list(self):
        cv_lists = self.data_dir_reader(self._txt_path)

        if self._load_type == 'train':
            final_list = []
            for cv_list in cv_lists:
                video_name = cv_list[0]
                label = cv_list[1]

                dir_path = os.path.join(self._root_dir, video_name)
                for file_list in os.listdir(dir_path):
                    final_list.append([file_list, label])

            self._data_list = final_list

        elif self._load_type == 'test':
            data_list = []
            for cv_list in cv_lists:
                video_name = cv_list[0]
                label = cv_list[1]

                dir_path = os.path.join(self._root_dir, video_name)
                for file_list in os.listdir(dir_path):
                    if not(re.split('[-.]+', file_list)[-2] == 'original'):
                        continue
                    data_list.append([file_list, label])

            self._data_list = data_list

        else:
            print("input correct train_test_type argm")
            raise ValueError

if __name__ == "__main__":
    data_root = "/home/jm/Two-stream_data/HMDB51/preprocess/frames"
    txt_path = "/home/jm/Two-stream_data/HMDB51"
    train_set = HMDB(data_root, txt_path, 'train', 1)
    test_set = HMDB(data_root, txt_path, 'test', 1)

#  A  A
# (‘ㅅ‘=)
# J.M.Seo