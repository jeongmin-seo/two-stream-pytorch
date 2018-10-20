import random
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch


class SpatialCubeDataset(Dataset):
    def __init__(self, dic, in_channel, root_dir, mode, model_mode, transform=None):
        # Generate a 16 Frame clip
        self.keys = list(dic.keys())
        self.values = list(dic.values())
        self.dic = dic
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode
        self.model_mode = model_mode
        self.in_channel = in_channel
        self.img_rows = 108# 224# 112
        self.img_cols = 108# 224# 112
        self.n_label = 51

    def reset_idx(self, _idx, _n_frame):
        if _idx > _n_frame:
            return self.reset_idx(_idx - _n_frame, _n_frame)
        else:
            return _idx

    def stack_frame(self, keys, _n_frame, _idx):
        video_path = os.path.join(self.root_dir, keys.split('-')[0])

        cube = torch.FloatTensor(3, self.in_channel,self.img_rows, self.img_cols)
        i = int(_idx)

        for j in range(self.in_channel):
            idx = self.reset_idx(i + j, _n_frame)
            # idx = i + j
            frame_idx = "image_%05d.jpg" % idx
            image = os.path.join(video_path, frame_idx)
            img = (Image.open(image))

            X = self.transform(img)
            cube[:, j, :, :] = X
            img.close()
        return cube

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        # print ('mode:',self.mode,'calling Dataset:__getitem__ @ idx=%d'%idx)
        cur_key = self.keys[idx]
        nb_frame = self.dic[cur_key][0]
        if self.mode == 'train':
            # self.clips_idx = random.randint(1, int(nb_frame - self.in_channel))
            if self.model_mode == 'tsn':
                self.clips_idx = [random.randint(1, int(nb_frame / 3)),
                                  random.randint(int(nb_frame / 3), int(2 * nb_frame / 3)),
                                  random.randint(int(2 * nb_frame / 3), int(nb_frame))]  # TSN
            else:
                self.clips_idx = random.randint(1, int(nb_frame))
            self.video = cur_key.split('/')[0]

        elif self.mode == 'val':
            split_key = cur_key.split('-')
            self.video = split_key[0]
            self.clips_idx = int(cur_key.split('-')[1])
        else:
            raise ValueError('There are only train and val mode')

        label = self.dic[cur_key][1]

        if self.model_mode == 'tsn' and self.mode == 'train':
            data = []
            for idx in self.clips_idx:
                data.append(self.stack_frame(cur_key, nb_frame, idx))
        else:
            data = self.stack_frame(cur_key, nb_frame, self.clips_idx)


        if self.mode == 'train':
            sample = (data, label)
        elif self.mode == 'val':
            sample = (self.video, data, label)
        else:
            raise ValueError('There are only train and val mode')
        return sample


class TemporalCubeDataset(SpatialCubeDataset):
    def __init__(self, dic, in_channel, root_dir, mode, transform=None):
        super().__init__(dic, in_channel, root_dir, mode, transform)
        self.img_rows = 68
        self.img_cols = 68

    def stack_frame(self, keys, _n_frame, _idx):
        video_path = os.path.join(self.root_dir, keys.split('-')[0])

        cube = torch.FloatTensor(2, self.in_channel, self.img_rows, self.img_cols)
        i = int(_idx)

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

            cube[0, j, :, :] = X
            cube[1, j, :, :] = Y

            imgX.close()
            imgY.close()

        return cube


class CubeDataLoader:
    def __init__(self, BATCH_SIZE, num_workers, in_channel, path, txt_path, split_num, mode):

        self.BATCH_SIZE = BATCH_SIZE
        self.num_workers = num_workers
        self.in_channel = in_channel
        self.data_path = path
        self.text_path = txt_path
        self.split_num = split_num
        self.mode = mode
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
        self.val_sample19()
        train_loader = self.train()
        val_loader = self.val()

        return train_loader, val_loader, self.test_video

    def val_sample19(self):
        self.dic_test_idx = {}
        for video in self.test_video:
            sampling_interval = int((self.test_video[video][0] - self.in_channel + 1) / 19)
            for index in range(19):
                clip_idx = index * sampling_interval
                key = video + '-' + str(clip_idx + 1)
                self.dic_test_idx[key] = self.test_video[video]

    def train(self):
        if self.mode == "spatial":
            training_set = SpatialCubeDataset(dic=self.train_video,
                                              in_channel=self.in_channel,
                                              root_dir=self.data_path,
                                              mode='train',
                                              transform=transforms.Compose([
                                                  transforms.Resize([112, 112]),
                                                  # transforms.Scale([118,118]),
                                                  # transforms.Scale([108,108]),
                                                  transforms.RandomCrop([108, 108]),
                                                  #transforms.RandomCrop(224),
                                                  # transforms.RandomHorizontalFlip(),
                                                  # transforms.Grayscale(),
                                                  transforms.ToTensor(),
                                                  # transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                  #                      std=[0.229, 0.224, 0.225])

                                              ]), model_mode='tsn')
            print('==> Training data :', len(training_set), ' videos', training_set[1][0][0].size())

        elif self.mode == "temporal":
            training_set = TemporalCubeDataset(dic=self.train_video,
                                               in_channel=self.in_channel,
                                               root_dir=self.data_path,
                                               mode='train',
                                               transform=transforms.Compose([
                                                   # transforms.Scale([68,68]),
                                                   transforms.Resize(68),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize(mean=(0.5,), std=(0.5,))
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
        if self.mode == "spatial":
            validation_set = SpatialCubeDataset(dic=self.dic_test_idx,
                                                in_channel=self.in_channel,
                                                root_dir=self.data_path,
                                                mode='val',
                                                transform=transforms.Compose([
                                                    # transforms.Scale([108,108]),
                                                    transforms.Resize([108,108]),
                                                    transforms.ToTensor(),
                                                    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                                ]), model_mode='tsn')
            print('==> Validation data :', len(validation_set), ' frames', validation_set[1][1].size())
            # print validation_set[1]

        elif self.mode == "temporal":
            validation_set = TemporalCubeDataset(dic=self.dic_test_idx,
                                                 in_channel=self.in_channel,
                                                 root_dir=self.data_path,
                                                 mode='val',
                                                 transform=transforms.Compose([
                                                     #transforms.Scale([68, 68]),
                                                     transforms.Resize(68),
                                                     transforms.ToTensor(),
                                                     transforms.Normalize(mean=(0.5,), std=(0.5,))
                                                 ]))
            print('==> Validation data :', len(validation_set), ' frames', validation_set[1][1].size())

        val_loader = DataLoader(
            dataset=validation_set,
            batch_size=self.BATCH_SIZE,
            shuffle=False,
            num_workers=self.num_workers)

        return val_loader


#  A  A
# (‘ㅅ‘=)
# J.M.Seo
