import random
import os
import re
import math
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class SpatialDataset(Dataset):

    def __init__(self, dic, root_dir, mode, transform=None):
        # Generate a 16 Frame clip
        self.keys = list(dic.keys())
        self.values = list(dic.values())
        self.dic = dic
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode
        # self.n_label = 51

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        # print ('mode:',self.mode,'calling Dataset:__getitem__ @ idx=%d'%idx)
        cur_key = self.keys[idx]
        if self.mode == 'train':
            # nb_frame = self.values[cur_key][0]
            nb_frame = self.dic[cur_key][0]
            self.clips_idx = random.randint(1, int(nb_frame))
            self.video = cur_key.split('/')[0]
        elif self.mode == 'val':
            split_key = cur_key.split('-')
            self.video = split_key[0]
            self.clips_idx = int(split_key[1])
        else:
            raise ValueError('There are only train and val mode')

        #label = self.values[cur_key][1]

        # label = onehot_encode(self.dic[cur_key][1], self.n_label)
        label = self.dic[cur_key][1]
        video_name = self.keys[idx].split('-')[0]
        data_root = os.path.join(self.root_dir, video_name)
        cur_data_list = os.listdir(data_root)

        data_path = os.path.join(data_root, cur_data_list[self.clips_idx-1])
        data = Image.open(data_path)

        if self.transform:
            data = self.transform(data)

        if self.mode == 'train':
            sample = (data, label)
        elif self.mode == 'val':
            sample = (self.video, data, label)
        else:
            raise ValueError('There are only train and val mode')
        return sample


class RepresentationDataset(SpatialDataset):

    def __init__(self, dic, root_dir, mode, transform=None):
        # super(SpatialDataset).__init__(dic, root_dir, mode, transform)
        super().__init__(dic, root_dir, mode, transform)
        self.values = self.reset_value()

    def __len__(self):
        return len(self.values)

    def __getitem__(self, idx):
        data_path = self.values[idx]
        data = Image.open(data_path)
        data = self.transform(data)

        return data, data_path

    def reset_value(self):
        result_list = []
        for path in self.keys:
            data_path = os.path.join(self.root_dir, path)
            for data_name in os.listdir(data_path):
                if not re.split("[.]+", data_name)[-1] == "jpg":
                    continue

                save_name = os.path.join(data_path, data_name)
                result_list.append(save_name)

        return result_list


class LoaderInit:
    def __init__(self, batch_size, num_workers, path, txt_path, split_num):

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_path = path
        self.text_path = txt_path
        self.split_num = split_num
        self.train_video, self.test_video = self.load_train_test_list()

    @staticmethod
    def read_text_file(file_path):
        tmp = {}
        f = open(file_path, 'r')
        for line in f.readlines():
            line = line.replace('\n', '')
            split_line = line.split(" ")
            tmp[split_line[0]] = [int(split_line[1]), int(
                split_line[2])]  # split[0] is video name and split[1] and [2] are frame num and class label

        return tmp

    def load_train_test_list(self):
        train_file_path = os.path.join(self.text_path, "train_split%d.txt" % self.split_num)
        test_file_path = os.path.join(self.text_path, "train_split%d.txt" % self.split_num)

        train_video = self.read_text_file(train_file_path)
        test_video = self.read_text_file(test_file_path)

        return train_video, test_video


class SpatialDataLoader(LoaderInit):
    def __init__(self, img_size, batch_size, num_workers, path, txt_path, split_num, is_ae=False):

        # super(LoaderInit, self).__init__(BATCH_SIZE, num_workers, path, txt_path, split_num)
        super().__init__(batch_size, num_workers, path, txt_path, split_num)
        # split the training and testing videos
        self.img_size = img_size
        self.is_ae = is_ae
        self.dic_test_idx = {}

    def run(self):
        # self.load_frame_count()
        # self.get_training_dic()
        self.val_sample19()
        train_loader = self.train()
        val_loader = self.val()

        return train_loader, val_loader, self.test_video

    def val_sample19(self):

        for video in self.test_video:
            # sampling_interval = int((self.test_video[video][0] - 10 + 1) / 19)
            sampling_interval = math.ceil((self.test_video[video][0] - 10 + 1) / 19)
            for index in range(19):
                clip_idx = index * sampling_interval
                key = video + '-' + str(clip_idx + 1)
                self.dic_test_idx[key] = self.test_video[video]

    def train(self):
        if not self.is_ae:
            training_set = SpatialDataset(dic=self.train_video,
                                          root_dir=self.data_path,
                                          mode='train',
                                          transform=transforms.Compose([
                                              transforms.Scale([256,256]),
                                              transforms.RandomCrop(self.img_size),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                          ]))
            print('==> Training data :', len(training_set), ' videos', training_set[1][0].size())

        else:
            training_set = SpatialDataset(dic=self.train_video,
                                          root_dir=self.data_path,
                                          mode='train',
                                          transform=transforms.Compose([
                                              transforms.Scale([64, 64]),
                                              # transforms.RandomCrop(224),
                                              # transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                   std=[0.229, 0.224, 0.225])
                                          ]))
            print('==> Training data :', len(training_set), ' videos', training_set[1][0].size())

        train_loader = DataLoader(
            dataset=training_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )

        return train_loader

    def val(self):
        validation_set = SpatialDataset(dic=self.dic_test_idx,
                                        root_dir=self.data_path,
                                        mode='val',
                                        transform=transforms.Compose([
                                            transforms.Resize([self.img_size, self.img_size]),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                        ]))
        print('==> Validation data :', len(validation_set), ' frames', validation_set[1][1].size())
        # print validation_set[1]

        val_loader = DataLoader(
            dataset=validation_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers)

        return val_loader


class RepresentationLoader(LoaderInit):
    def __init__(self, batch_size, num_workers, path, txt_path, split_num):
        # super(LoaderInit, self).__init__(BATCH_SIZE, num_workers, path, txt_path, split_num)
        super().__init__(batch_size, num_workers, path, txt_path, split_num)

    def run(self):
        represent_set = RepresentationDataset(dic=self.train_video,
                                              root_dir=self.data_path,
                                              mode='train',
                                              transform=transforms.Compose([
                                                  transforms.Scale([64, 64]),
                                                  # transforms.RandomCrop(224),
                                                  # transforms.RandomHorizontalFlip(),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                       std=[0.229, 0.224, 0.225])
                                              ]))
        loader = DataLoader(
            dataset=represent_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
        return loader
#  A  A
# (‘ㅅ‘=)
# J.M.Seo
