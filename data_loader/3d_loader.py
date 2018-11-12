import random
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch


class OneCubeDataset(Dataset):
    def __init__(self, dic, in_channel, root_dir, mode, transform=None):
        # Generate a 16 Frame clip
        self.keys = list(dic.keys())
        self.values = list(dic.values())
        self.dic = dic
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode
        self.in_channel = in_channel
        self.img_rows = 112
        self.img_cols = 112
        self.n_label = 51

    def reset_idx(self, _idx, _n_frame):
        if _idx > _n_frame:
            return self.reset_idx(_idx - _n_frame, _n_frame)
        else:
            return _idx

    def stack_frame(self, keys, _n_frame, _step):
        video_path = os.path.join(self.root_dir, keys.split('-')[0])

        cube = torch.FloatTensor(3, self.in_channel,self.img_rows, self.img_cols)

        for j in range(self.in_channel):
            idx = self.reset_idx(j * _step + 1, _n_frame)
            frame_idx = "image_%05d.jpg" % idx
            image = os.path.join(video_path, frame_idx)
            img = (Image.open(image))

            X = self.transform(img)
            cube[:, j, :, :] = X
            img.close()
        return cube

    def get_step_size(self, _nb_frame):
        if _nb_frame <= self.in_channel:
            return 1
        else:
            return int(_nb_frame/self.in_channel)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        # print ('mode:',self.mode,'calling Dataset:__getitem__ @ idx=%d'%idx)
        cur_key = self.keys[idx]
        nb_frame = self.dic[cur_key][0]

        label = self.dic[cur_key][1]
        step = self.get_step_size(nb_frame)
        data = self.stack_frame(cur_key, nb_frame, step)

        sample = (data, label)

        return sample


class CubeDataLoader:
    def __init__(self, BATCH_SIZE, num_workers, in_channel, path, txt_path, split_num):

        self.BATCH_SIZE = BATCH_SIZE
        self.num_workers = num_workers
        self.in_channel = in_channel
        self.data_path = path
        self.text_path = txt_path
        self.split_num = split_num

        # split the training and testing videos
        self.train_video, self.test_video = self.load_train_test_list()

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
        test_file_path = os.path.join(self.text_path, "test_split%d.txt" % self.split_num)

        train_video = self.read_text_file(train_file_path)
        test_video = self.read_text_file(test_file_path)

        return train_video, test_video

    def run(self):
        train_loader = self.train()
        val_loader = self.val()

        return train_loader, val_loader, self.test_video



    def train(self):

        training_set = OneCubeDataset(dic=self.train_video,
                                      in_channel=self.in_channel,
                                      root_dir=self.data_path,
                                      mode='train',
                                      transform=transforms.Compose([
                                          transforms.Resize([224, 224]),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.5,], std=[0.5,])
                                      ]))
        print('==> Training data :', len(training_set), ' videos', training_set[1][0][0].size())

        train_loader = DataLoader(
            dataset=training_set,
            batch_size=self.BATCH_SIZE,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

        return train_loader

    def val(self):

        validation_set = OneCubeDataset(dic=self.test_video,
                                        in_channel=self.in_channel,
                                        root_dir=self.data_path,
                                        mode='val',
                                        transform=transforms.Compose([
                                            transforms.Resize([224,224]),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.5,], std=[0.5,])
                                        ]))
        print('==> Validation data :', len(validation_set), ' frames', validation_set[1][1].size())
        # print validation_set[1]



        val_loader = DataLoader(
            dataset=validation_set,
            batch_size=self.BATCH_SIZE,
            shuffle=False,
            num_workers=self.num_workers)

        return val_loader


#  A  A
# (‘ㅅ‘=)
# J.M.Seo
