import random
import os
import cv2


class spatial_dataset(Dataset):
    def __init__(self, dic, in_channel, root_dir, mode, transform=None):
        # Generate a 16 Frame clip
        self.keys = dic.keys()
        self.values = dic.values()
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode
        self.in_channel = in_channel
        self.img_rows = 224
        self.img_cols = 224

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        # print ('mode:',self.mode,'calling Dataset:__getitem__ @ idx=%d'%idx)
        cur_key = self.keys[idx]
        if self.mode == 'train':
            nb_frame = self.values[cur_key][0]
            self.clips_idx = random.randint(1, int(nb_frame))
        elif self.mode == 'val':
            nb_frame = self.values[cur_key][0]
            self.video = cur_key.split('/')[0]
        else:
            raise ValueError('There are only train and val mode')

        label = self.values[cur_key][1]
        data_root = os.path.join(self.root_dir, self.keys[idx])
        cur_data_list = os.listdir(data_root)
        data_path = os.path.join(data_root, cur_data_list[self.clips_idx-1])
        data = cv2.imread(data_path)

        if self.mode == 'train':
            sample = (data, label)
        elif self.mode == 'val':
            sample = (self.video, data, label)
        else:
            raise ValueError('There are only train and val mode')
        return sample


class Spatial_DataLoader():
    def __init__(self, BATCH_SIZE, num_workers, in_channel, path, txt_path, split_num):

        self.BATCH_SIZE = BATCH_SIZE
        self.num_workers = num_workers
        self.frame_count = {}
        self.in_channel = in_channel
        self.data_path = path
        self.text_path = txt_path
        self.split_num = split_num
        # split the training and testing videos
        splitter = UCF101_splitter(path=ucf_list, split=ucf_split)
        self.train_video, self.test_video = splitter.split_video()

    @staticmethod
    def read_text_file(file_path):
        f = open(file_path, 'r')
        for line in f.readlines():
            split_line = line.split

    def load_train_test_list(self):
        test_file_path = os.path.join(self.text_path, "test_split%d.txt" % self.split_num)
        train_file_path = os.path.join(self.text_path, "train_split%d.txt" % self.split_num)

    def load_frame_count(self):
        # print '==> Loading frame number of each video'
        with open('dic/frame_count.pickle', 'rb') as file:
            dic_frame = pickle.load(file)
        file.close()

        for line in dic_frame:
            videoname = line.split('_', 1)[1].split('.', 1)[0]
            n, g = videoname.split('_', 1)
            if n == 'HandStandPushups':
                videoname = 'HandstandPushups_' + g
            self.frame_count[videoname] = dic_frame[line]

    def run(self):
        self.load_frame_count()
        self.get_training_dic()
        self.val_sample19()
        train_loader = self.train()
        val_loader = self.val()

        return train_loader, val_loader, self.test_video

    def val_sample19(self):
        self.dic_test_idx = {}
        # print len(self.test_video)
        for video in self.test_video:
            n, g = video.split('_', 1)

            sampling_interval = int((self.frame_count[video] - 10 + 1) / 19)
            for index in range(19):
                clip_idx = index * sampling_interval
                key = video + '-' + str(clip_idx + 1)
                self.dic_test_idx[key] = self.test_video[video]

    def get_training_dic(self):
        self.dic_video_train = {}
        for video in self.train_video:
            nb_clips = self.frame_count[video] - 10 + 1
            key = video + '-' + str(nb_clips)
            self.dic_video_train[key] = self.train_video[video]

    def train(self):
        training_set = motion_dataset(dic=self.dic_video_train, in_channel=self.in_channel, root_dir=self.data_path,
                                      mode='train',
                                      transform=transforms.Compose([
                                          transforms.Scale([224, 224]),
                                          transforms.ToTensor(),
                                      ]))
        print
        '==> Training data :', len(training_set), ' videos', training_set[1][0].size()

        train_loader = DataLoader(
            dataset=training_set,
            batch_size=self.BATCH_SIZE,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

        return train_loader

    def val(self):
        validation_set = motion_dataset(dic=self.dic_test_idx, in_channel=self.in_channel, root_dir=self.data_path,
                                        mode='val',
                                        transform=transforms.Compose([
                                            transforms.Scale([224, 224]),
                                            transforms.ToTensor(),
                                        ]))
        print
        '==> Validation data :', len(validation_set), ' frames', validation_set[1][1].size()
        # print validation_set[1]

        val_loader = DataLoader(
            dataset=validation_set,
            batch_size=self.BATCH_SIZE,
            shuffle=False,
            num_workers=self.num_workers)

        return val_loader