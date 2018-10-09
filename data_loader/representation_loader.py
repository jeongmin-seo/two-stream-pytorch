import os
import re
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import torchvision.transforms as transforms

class RepresentationDataset(Dataset):

    def __init__(self, video_dic, root_dir):
        self.v_list = list(video_dic.keys())
        self.video_info = video_dic
        self.data_root = root_dir

    def __len__(self):
        return len(self.v_list)

    def __getitem__(self, idx):
        video_root = self.v_list[idx]
        cur_video_info = self.video_info[video_root]
        video_root = os.path.join(self.data_root, video_root)
        file_list = sorted(os.listdir(video_root))

        """
        for file_name in file_list:
            if not re.split("[.]+", file_name)[-1] == "npy":
                continue

            if int(re.split("[_.]+", file_name)[-2]) == 1:
                dat = np.load(os.path.join(video_root, file_name))

            else:
                dat = np.hstack((dat, np.load(os.path.join(video_root, file_name))))
        """
        dat = torch.FloatTensor(cur_video_info[0], 2000)
        for i in range(1, cur_video_info[0]+1):
            file_name = "image_%05d.npy" % i

            dat[i-1, :] = torch.from_numpy(np.load(os.path.join(video_root, file_name)))
            """
            if i == 1:
                dat = np.load(os.path.join(video_root, file_name))
                dat[i-1,:] = np.load(os.path.join(video_root, file_name))

            else:
                # dat = np.hstack((dat, np.load(os.path.join(video_root, file_name))))
                dat = np.row_stack((dat, np.load(os.path.join(video_root, file_name))))
            """
        label = cur_video_info[1]
        return dat, label, video_root



class RepresentationLoader:

    def __init__(self, batch_size, num_workers, path, txt_path, split_num):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_path = path
        self.text_path = txt_path
        self.split_num = split_num
        self.train_video = self.load_train_test_list()

    @staticmethod
    def read_text_file(file_path):
        tmp = {}
        f = open(file_path, 'r')
        for line in f.readlines():
            line = line.replace('\n', '')
            split_line = line.split(" ")
            tmp[split_line[0]] = [int(split_line[1]), int(split_line[2])]  # split[0] is video name and split[1] and [2] are frame num and class label

        return tmp

    def load_train_test_list(self):
        train_file_path = os.path.join(self.text_path, "train_split%d.txt" % self.split_num)
        # test_file_path = os.path.join(self.text_path, "test_split%d.txt" % self.split_num)

        train_video = self.read_text_file(train_file_path)
        # test_video = self.read_text_file(test_file_path)

        return train_video# , test_video

    def run(self):
        training_set = RepresentationDataset(self.train_video,
                                             self.data_path)
        train_loader = DataLoader(training_set, batch_size=self.batch_size, shuffle=True)
        return train_loader


if __name__=="__main__":

    """
    root = "/home/jm/hdd/representation/split1"

    for action_list in os.listdir(root):
        action_root = os.path.join(root, action_list)
        for v_list in os.listdir(action_list):
            video_root = os.path.join(action_root, v_list)
            for img_list in os.listdir(video_root):
                dat_path = os.path.join(video_root, img_list)

                np.load(dat_path)
    """

#  A  A
# (‘ㅅ‘=)
# J.M.Seo