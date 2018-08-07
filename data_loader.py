########################################
#     import requirement libraries     #
########################################
import os
import re
import numpy as np
import random
from torch.utils.data import DataLoader

DataLoader()
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