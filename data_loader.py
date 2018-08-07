########################################
#     import requirement libraries     #
########################################
import os
import re
import numpy as np
import random


from keras.utils import to_categorical
li = []


class DataLoader:
    def __init__(self, root_dir, batch_size=32):
        self._train_data_list = []
        self._test_data = {}
        self._test_list = []
        self._root_dir = root_dir   # data root
        self._batch_size = batch_size
        self._front_idx = 0
        self._end_idx = batch_size

    def set_data_list(self, _data_txt, train_test_type='train'):
        cv_lists = self.data_dir_reader(_data_txt)

        if train_test_type == 'train':
            final_list = []
            for cv_list in cv_lists:
                video_name = cv_list[0]
                label = cv_list[1]

                dir_path = os.path.join(self._root_dir, video_name)
                for file_list in os.listdir(dir_path):
                    final_list.append([file_list, label])

            self._train_data_list = final_list

        elif train_test_type == 'test':

            for cv_list in cv_lists:
                video_name = cv_list[0]
                label = cv_list[1]

                data_list = []
                dir_path = os.path.join(self._root_dir, video_name)
                for file_list in os.listdir(dir_path):
                    if not(re.split('[-.]+', file_list)[-2] == 'original'):
                        continue

                    data_list.append([file_list, label])

                self._test_data[video_name] = data_list

        else:
            print("input correct train_test_type argm")
            raise ValueError

    @staticmethod
    def data_dir_reader(_txt_path):

        tmp = []
        f = open(_txt_path, 'r')
        for line in f.readlines():
            line = line.replace('\n', '')
            split_line = re.split(r"[\s+,\/]*", line)
            # split_info = dir_name.split(' ')
            # split_info[0] = split_info[0].replace('/', '-')
            file_info = split_line[0] + "-%05d" % int(split_line[1])
            tmp.append([file_info, int(split_line[-1])])

        return tmp

    def next_test_video(self):
        data = []
        label = []
        end_of_file = False

        test_data_list = self._test_data[self._test_list.pop()]
        for data_list in test_data_list:
            file_name = data_list[0]
            cur_label = data_list[1]

            split_file_name = re.split('[-.]+', file_name)
            file_dir_name = split_file_name[0] + '-' + split_file_name[1]
            file_root = os.path.join(self._root_dir, file_dir_name, file_name)

            try:
                dat = np.load(file_root)
                data.append(dat)
                onehot_label = to_categorical(cur_label, num_classes=51)
                label.append(onehot_label)
                # TODO: make label using video_name

            except FileNotFoundError:
                continue

        if not self._test_list:
            end_of_file = True

        return np.asarray(data), np.asarray(label), end_of_file

    def next_train_batch(self):

        data = []
        label = []
        end_of_file = False

        for data_list in self._train_data_list[self._front_idx: self._end_idx]:
            file_name = data_list[0]
            cur_label = data_list[1]

            split_file_name = re.split('[-.]+', file_name)
            file_dir_name = split_file_name[0] + '-' + split_file_name[1]
            file_root = os.path.join(self._root_dir, file_dir_name, file_name)

            try:
                dat = np.load(file_root)
                data.append(dat)

                onehot_label = to_categorical(cur_label, num_classes=51)
                label.append(onehot_label)
                # TODO: make label using video_name

            except FileNotFoundError:
                continue

        self._front_idx += self._batch_size
        self._end_idx += self._batch_size

        if self._end_idx > len(self._train_data_list):
            self._end_idx = len(self._train_data_list)

        if self._front_idx > len(self._train_data_list):
            end_of_file = True
            self._front_idx = 0
            self._end_idx = self._batch_size

        if not data:
            print("Data is empty!!")

        if not label:
            print('Label is empty!!')

        return np.asarray(data), np.asarray(label), end_of_file

    def train_data_shuffle(self):

        if not self._train_data_list:
            print('Data list is None')
            raise IndexError

        random.shuffle(self._train_data_list)
        self._front_idx = 0
        self._end_idx = self._batch_size

    def set_test_video_list(self):
        self._test_list = list(self._test_data.keys())

    def get_train_data_list(self):
        return self._train_data_list

    def get_test_data_list(self):
        return self._test_list

if __name__ == '__main__':

    # HMDB-51 data loader
    root = '/home/jm/hdd/preprocess/frames'
    txt_root = '/home/jm/Two-stream_data/HMDB51/train_split1.txt'

    train_loader = DataLoader(root)
    train_loader.set_data_list(txt_root, 'train')

    n = 0
    for epoch in range(1):
        train_loader.train_data_shuffle()
        while True:
            x, y, eof = train_loader.next_train_batch()

            print(x.shape)
            n += 1

            if eof:
                break

    print("*"*50)
    txt_root = '/home/jm/Two-stream_data/HMDB51/test_split1.txt'
    test_loader = DataLoader(root)
    test_loader.set_data_list(txt_root, 'test')

    n=0
    for epoch in range(5):
        test_loader.set_test_video_list()
        while True:

            x, y, eof = test_loader.next_test_video()

            print(x.shape)
            n += 1

            if eof:
                break

#  A  A
# (‘ㅅ‘=)
# J.M.Seo