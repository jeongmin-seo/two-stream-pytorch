import random
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import torch


class MotionDataset(Dataset):
    def __init__(self, dic, img_size, in_channel, root_dir, mode, transform=None):
        # Generate a 16 Frame clip
        self.keys = list(dic.keys())
        self.values = list(dic.values())
        self.dic = dic
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode
        self.img_size = img_size
        self.in_channel = in_channel

    def stackopf(self, keys):
        video_path = os.path.join(self.root_dir, keys.split('-')[0])

        flow = torch.FloatTensor(2 * self.in_channel, self.img_size, self.img_size)
        i = int(self.clips_idx)

        for j in range(self.in_channel):
            idx = i + j
            frame_idx = "%05d.jpg" % idx
            # frame_idx = 'frame' + idx.zfill(6)
            x_image = os.path.join(video_path, 'flow_x_' + frame_idx)
            y_image = os.path.join(video_path, 'flow_y_' + frame_idx)

            imgX = (Image.open(x_image))
            imgY = (Image.open(y_image))

            X = self.transform(imgX)
            Y = self.transform(imgY)

            flow[2 * (j - 1), :, :] = X
            flow[2 * (j - 1) + 1, :, :] = Y
            imgX.close()
            imgY.close()
        return flow

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        # print ('mode:',self.mode,'calling Dataset:__getitem__ @ idx=%d'%idx)
        cur_key = self.keys[idx]
        if self.mode == 'train':
            # nb_frame = self.values[cur_key][0]
            nb_frame = self.dic[cur_key][0]
            self.clips_idx = random.randint(1, int(nb_frame-self.in_channel+1))
            self.video = cur_key.split('/')[0]
        elif self.mode == 'val':
            split_key = cur_key.split('-')
            self.video = split_key[0]
            self.clips_idx = int(cur_key.split('-')[1])
        else:
            raise ValueError('There are only train and val mode')

        # label = self.values[cur_key][1]
        label = self.dic[cur_key][1]
        data = self.stackopf(cur_key)

        if self.mode == 'train':
            sample = (data, label)
        elif self.mode == 'val':
            sample = (self.video, data, label)
        else:
            raise ValueError('There are only train and val mode')
        return sample


class MotionDataLoader():
    def __init__(self, img_size, batch_size, num_workers, in_channel, path, txt_path, split_num):

        self.img_size = img_size
        self.BATCH_SIZE = batch_size
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
            tmp[split_line[0]] = [int(split_line[1]) - 1, int(split_line[2])]  # split[0] is video name and split[1] and [2] are frame num and class label

        return tmp

    def load_train_test_list(self):
        train_file_path = os.path.join(self.text_path, "train_split%d.txt" % self.split_num)
        test_file_path = os.path.join(self.text_path, "test_split%d.txt" % self.split_num)

        train_video = self.read_text_file(train_file_path)
        test_video = self.read_text_file(test_file_path)

        return train_video, test_video

    def run(self):
        self.val_sample19()
        train_loader = self.train()
        val_loader = self.val()

        return train_loader, val_loader, self.test_video

    def val_sample19(self):
        self.dic_test_idx = {}
        for video in self.test_video:
            # n, g = video.split('_', 1)

            sampling_interval = int((self.test_video[video][0] - 10 + 1) / 19)
            for index in range(19):
                clip_idx = index * sampling_interval
                key = video + '-' + str(clip_idx + 1)
                self.dic_test_idx[key] = self.test_video[video]

    def train(self):
        training_set = MotionDataset(dic=self.train_video, in_channel=self.in_channel, root_dir=self.data_path,
                                     mode='train', img_size=self.img_size,
                                     transform=transforms.Compose([
                                         transforms.Resize([self.img_size, self.img_size]),
                                         transforms.ToTensor()
                                     ]))
        print('==> Training data :', len(training_set), ' videos', training_set[1][0].size())

        train_loader = DataLoader(
            dataset=training_set,
            batch_size=self.BATCH_SIZE,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

        return train_loader

    def val(self):
        validation_set = MotionDataset(dic=self.dic_test_idx, in_channel=self.in_channel, root_dir=self.data_path,
                                       mode='val', img_size=self.img_size,
                                       transform=transforms.Compose([
                                           transforms.Resize([self.img_size, self.img_size]),
                                           transforms.ToTensor()
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