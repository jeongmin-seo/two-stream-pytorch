import random
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class JHMDBDataset(Dataset):
    def __init__(self, data_root, txt_file_path, L, img_rows, img_cols, transform):
        self.data_root = data_root
        self.txt_path = txt_file_path
        self.L = L
        self.data_list = self.txt_file_reader()
        self.keys = self.data_list.keys()
        self.transform = transform
        self.img_rows = 112
        self.img_cols = 112

    def __getitem__(self, idx):
        file_path = self.keys[0]
        n_frame = self.data_list[file_path][0]
        class_info = self.data_list[file_path][1]

        cube = torch.FloatTensor(1, self.L, self.img_rows, self.img_cols)
        clips_idx = random.randint(1, int(n_frame - self.L + 1))
        video_path = os.path.join(self.data_root, file_path)
        for i in range(self.L):
            idx = clips_idx + i
            frame_idx = "%05d.png" % idx
            image = os.path.join(video_path, frame_idx)
            img = (Image.open(image))

            X = self.transform(img)
            cube[:, i, :, :] = X
            img.close()

        return cube , class_info

    def txt_file_reader(self):

        if not self.txt_path:
            raise FileNotFoundError

        tmp = {}
        f = open(self.txt_path, 'r')
        for line in f.readlines():
            split_line = line.split(' ')
            tmp[split_line[0]] = [int(split_line[1]), int(split_line[2])]

        return tmp

class JHMDBLoader(DataLoader):
    def __init__(self, batch_size, ):
        pass
