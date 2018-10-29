import os
import re
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
import torchvision.transforms as transforms


class DCNNDataset(Dataset):

    def __init__(self, dic, root_dir, max_frame_num, transform=None):
        self.keys = list(dic.keys())
        self.dic = dic
        self.root_dir = root_dir
        self.transform = transform
        self.max_frame_num = max_frame_num
        self.img_size = 224

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        cur_video_key = self.keys[idx]
        cur_video_path = os.path.join(self.root_dir, cur_video_key)
        nb_frame = self.dic[cur_video_key][0]
        label = int(self.dic[cur_video_key][1])

        cube = torch.FloatTensor(3, self.max_frame_num, self.img_size, self.img_size)

        j = 0
        for file_name in sorted(os.listdir(cur_video_path)):
            if not re.split('[.]+', file_name)[-1] == 'jpg':
                continue

            img = (Image.open(os.path.join(cur_video_path, file_name)))
            X = self.transform(img)
            cube[:, j, :, :] = X
            j = j + 1
            img.close()

        return cube, label, cur_video_key


class DCNNLoader:

    def __init__(self, batch_size, num_workers, path, txt_path, split_num, max_frame_num):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_path = path
        self.text_path = txt_path
        self.split_num = split_num
        self.max_frame_num = max_frame_num

        # split the training and testing videos
        self.train_video, self.test_video = self.load_train_test_list()

    @staticmethod
    def read_text_file(file_path):
        tmp = {}
        f = open(file_path, 'r')
        for line in f.readlines():
            line = line.replace('\n', '')
            split_line = line.split(" ")

            # split[0] is video name and split[1] and [2] are frame num and class label
            tmp[split_line[0]] = [int(split_line[1]), int(split_line[2])]

        return tmp

    def load_train_test_list(self):
        train_file_path = os.path.join(self.text_path, "train_split%d.txt" % self.split_num)
        test_file_path = os.path.join(self.text_path, "test_split%d.txt" % self.split_num)

        train_video = self.read_text_file(train_file_path)
        test_video = self.read_text_file(test_file_path)

        return train_video, test_video

    def run(self):
        training_set = DCNNDataset(self.train_video,
                                   self.data_path,
                                   self.max_frame_num,
                                   transform=transforms.Compose([
                                       transforms.Scale([224, 224]),
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                            std=[0.229, 0.224, 0.225])
                                   ]))
        train_loader = DataLoader(training_set, batch_size=self.batch_size, shuffle=False)

        validation_set = DCNNDataset(self.test_video,
                                     self.data_path,
                                     self.max_frame_num,
                                     transform=transforms.Compose([
                                         transforms.Scale([224,224]),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                              std=[0.229, 0.224, 0.225])
                                     ]))
        val_loader = DataLoader(validation_set, batch_size=self.batch_size, shuffle=False)
        return train_loader, val_loader


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