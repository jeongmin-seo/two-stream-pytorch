import os
import re
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
import torchvision.transforms as transforms

class RepresentationDataset(Dataset):

    def __init__(self, all_frame_path, transform=None):
        self.frame_path = all_frame_path
        self.transform = transform

    def __len__(self):
        return len(self.frame_path)

    def __getitem__(self, idx):
        file_path = self.frame_path[idx]
        img = Image.open(file_path)
        img = self.transform(img)

        return img, file_path



class RepresentationLoader:

    def __init__(self, batch_size, num_workers, path):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_path = path

        self.all_frame_path = self.load_all_file_path()

    def load_all_file_path(self):

        result = []
        for action in os.listdir(self.data_path):
            vid_root = os.path.join(self.data_path, action)

            for vid_num in os.listdir(vid_root):
                frame_root = os.path.join(vid_root, vid_num)

                for frame_name in os.listdir(frame_root):
                    if not re.split('[.]+', frame_name)[-1] == 'jpg':
                        continue
                    result.append(os.path.join(frame_root, frame_name))
        return result

    def run(self):
        training_set = RepresentationDataset(self.all_frame_path,
                                             transform=transforms.Compose([
                                                 transforms.Scale([64, 64]),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                      std=[0.229, 0.224, 0.225])
                                             ]))
        loader = DataLoader(training_set, batch_size=self.batch_size, shuffle=False)
        return loader


if __name__=="__main__":

    """
    root = "/home/jm/hdd/representation/split1"

    for action_list in os.listdir(root):
        action_root = os.path.join(root, action_list)
        for v_list in os.listdir(action_list):
            video_root = os.path.join(action_root, v_list)
            for img_list in os.listdir(video_root):
                dat_path = os.path.join(video_root, img_list)

                np.load(dat_path)
    """

#  A  A
# (‘ㅅ‘=)
# J.M.Seo