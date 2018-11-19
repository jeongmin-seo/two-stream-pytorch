import random
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch




class TemporalCubeDataset(Dataset):
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
        # self.n_label = 51

    def stack_frame(self, keys, _n_frame, _idx):
        video_path = os.path.join(self.root_dir, keys.split('-')[0])

        cube = torch.FloatTensor(2, self.in_channel, self.img_size, self.img_size)
        i = int(_idx)

        for j in range(self.in_channel):
            idx = self.reset_idx(i + j, _n_frame)
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

    def reset_idx(self, _idx, _n_frame):
        if _idx > _n_frame:
            return self.reset_idx(_idx - _n_frame, _n_frame)
        else:
            return _idx

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        # print ('mode:',self.mode,'calling Dataset:__getitem__ @ idx=%d'%idx)
        cur_key = self.keys[idx]
        nb_frame = self.dic[cur_key][0]
        label = self.dic[cur_key][1]

        self.clips_idx = random.randint(1, int(nb_frame))
        self.video = cur_key.split('/')[0]
        data = self.stack_frame(cur_key, nb_frame, self.clips_idx)
        sample = (data, label)

        return sample


class CubeDataLoader:
    def __init__(self, img_size, batch_size, num_workers, in_channel, path):

        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.in_channel = in_channel
        self.data_path = path
        self.label = sorted(os.listdir(self.data_path))
        self.train_video = self.read_all_data()

    def read_all_data(self):
        train_video = dict()
        for action in self.label:
            video_root = os.path.join(self.data_path, action)
            for video_name in os.listdir(video_root):
                key = action + '/' + video_name
                train_video[key] = [int(len(os.listdir(os.path.join(video_root, video_name)))/2),self.label.index(action)]

        return train_video

    def train(self):

        training_set = TemporalCubeDataset(dic=self.train_video,
                                           img_size = self.img_size,
                                           in_channel=self.in_channel,
                                           root_dir=self.data_path,
                                           mode='train',
                                           transform=transforms.Compose([
                                               # transforms.Scale([68,68]),
                                               transforms.Resize([self.img_size, self.img_size]),
                                               transforms.ToTensor(),
                                               transforms.Normalize(mean=(0.5,), std=(0.5,))
                                           ]))
        print('==> Training data :', len(training_set), ' videos', training_set[0][0].size())

        train_loader = DataLoader(
            dataset=training_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

        return train_loader



#  A  A
# (‘ㅅ‘=)
# J.M.Seo
