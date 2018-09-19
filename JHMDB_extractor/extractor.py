import numpy as np
import os
import pickle
import cv2
import re
import scipy.io as sio

data_root = "/home/jm/hdd/JHMDB"
mask_root = os.path.join(data_root, "puppet_mask")
image_root = os.path.join(data_root, "Rename_Images")
save_root = os.path.join(data_root, "background_extract_image")

if not os.path.isdir(save_root):
    os.mkdir(save_root)

def read_mat_file(_file_root):
    file_name = os.path.join(_file_root, "puppet_mask.mat")
    return sio.loadmat(file_name)['part_mask']

def delete_background(_mask, _img_root, _save_path):

    for file_name in sorted(os.listdir(_img_root)):
        split_name = re.split("[.]+", file_name)
        if split_name[-1] != "png":
            continue

        if _mask.shape[2] <= int(split_name[0])-1 :
            continue

        load_img_name = os.path.join(_img_root, file_name)
        save_file_name = os.path.join(_save_path, file_name)

        cur_mask = _mask[:,:, int(split_name[0])-1]
        img = cv2.imread(load_img_name, cv2.IMREAD_GRAYSCALE)

        extract_img = img * cur_mask
        cv2.imwrite(save_file_name, extract_img)

    return True

if __name__=="__main__":

    """
    mask_path = "/home/jm/hdd/JHMDB/puppet_mask/stand/BIG_FISH_stand_u_nm_np1_fr_med_27"
    mask = read_mat_file(mask_path)
    print(mask.shape)
    # delete_background(mask, )
    """

    for class_name in os.listdir(mask_root):

        save_class_dir = os.path.join(save_root, class_name)
        img_class_dir = os.path.join(image_root, class_name)
        mask_class_dir = os.path.join(mask_root, class_name)

        if not os.path.isdir(save_class_dir):
            os.mkdir(save_class_dir)

        for video_name in os.listdir(mask_class_dir):
            print(class_name + "/" + video_name)

            save_video_root = os.path.join(save_class_dir, video_name)
            img_video_root = os.path.join(img_class_dir, video_name)
            mask_video_root = os.path.join(mask_class_dir, video_name)

            if not os.path.isdir(save_video_root):
                os.mkdir(save_video_root)

            mask = read_mat_file(mask_video_root)
            delete_background(mask, img_video_root, save_video_root)



