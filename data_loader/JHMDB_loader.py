import random
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class JHMDBDataset(Dataset):
    def __init__(self, dic, data_root, L, transform, img_size=(112,112)):
        self.data_root = data_root
        self.L = L
        self.dic = dic
        self.keys = dic.keys()
        self.transform = transform
        self.img_rows = img_size[0]
        self.img_cols = img_size[1]

    def __getitem__(self, idx):
        file_path = self.keys[0]
        n_frame = self.dic[file_path][0]
        class_info = self.dic[file_path][1]

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

class JHMDBLoader:
    def __init__(self, batch_size, num_workers, in_channel, path, txt_path, img_size=(112,112)):

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.in_channel = in_channel
        self.data_path = path
        self.txt_path = txt_path
        self.img_size = img_size
        # split the training and testing videos
        self.video_info = self.txt_file_reader()

    def txt_file_reader(self):

        if not self.txt_path:
            raise FileNotFoundError

        tmp = {}
        f = open(self.txt_path, 'r')
        for line in f.readlines():
            split_line = line.split(' ')
            tmp[split_line[0]] = [int(split_line[1]), int(split_line[2])]

        return tmp

    def load(self):
        jhmdb_set = JHMDBDataset(dic=self.video_info,
                                 data_root=self.data_path,
                                 L  =self.in_channel,
                                 transform=transforms.Compose([
                                     transforms.Scale([self.img_size[0],self.img_size[1]]),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                          std=[0.229, 0.224, 0.225])
                                 ]))
        print('==> Data :', len(jhmdb_set), ' videos', jhmdb_set[1][0].size())

        data_loader = DataLoader(dataset=jhmdb_set,
                                 batch_size=self.batch_size,
                                 shuffle=True,
                                 num_workers=self.num_workers,
                                 pin_memory=True)

        return data_loader