# -*- coding: utf-8 -*-
import os
import pickle
import argparse
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
import cv2

parser = argparse.ArgumentParser(description='PyTorch Two-Stream Action Recognition')
parser.add_argument('--video', metavar='PATH', type=str, help='test video path')
parser.add_argument('--ext_path', type=str, help="Savepath for extracted frame and flow")
parser.add_argument('--model_root', type=str, help='test model root')
parser.add_argument('--model', type=str, help='selected model name')
parser.add_argument('--fusion_rate', type=str)

def make_dir(modal, result_path, vid_name):
    modal_root = os.path.join(result_path, vid_name)
    if not os.path.isdir(modal_root):
        os.mkdir(modal_root)

    modal_path = os.path.join(modal_root, modal)
    if not os.path.isdir(modal_path):
        os.mkdir(modal_path)

    return modal_path

def extract_data(video_path, result_path):
    vid_name = video_path.split('/')[-1].split('.')[0:-1][0]
    frame_root = make_dir('frames', result_path, vid_name)
    flow_root = make_dir('flow', result_path, vid_name)

    cap = cv2.VideoCapture(video_path)

    frame_idx = 1
    ret, old_frame = cap.read()
    prvs = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    frame_rgb = cv2.cvtColor(old_frame, cv2.COLOR_BGR2RGB)
    cv2.imwrite(os.path.join(frame_root, "image_%05d.jpg" % frame_idx), frame_rgb)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx = frame_idx + 1
        next = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flow_x = normalize_flow(flow[...,0])
        flow_y = normalize_flow(flow[...,1])

        cv2.imwrite(os.path.join(frame_root, "image_%05d.jpg" % frame_idx), frame)
        cv2.imwrite(os.path.join(flow_root, "flow_x_%05d.jpg" % (frame_idx-1)), flow_x)
        cv2.imwrite(os.path.join(flow_root, "flow_y_%05d.jpg" % (frame_idx - 1)), flow_y)

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        prvs = next

    cap.release()
    cv2.destroyAllWindows()

    return vid_name

def normalize_flow(target_flow):
    target_flow[target_flow < -20] = -20
    target_flow[target_flow > 20] = 20
    target_flow = target_flow * (255/40) + (255/2)

    return np.floor(target_flow).astype(np.uint8)



def set_transforms(mode):
    if mode == 'spatial':
        transform = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    elif mode == 'temporal':
        transform = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor()])

    else:
        transform = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    return transform

def val_sample19(n_frame, _in_channel):
    sampled_idx = list()
    sampling_interval = int((n_frame - _in_channel +1) / 19)

    for index in range(19):
        clip_idx = index * sampling_interval + 1
        sampled_idx.append(clip_idx)

    return sampled_idx

def reset_idx(_idx, _n_frame):
    if _idx > _n_frame:
        return reset_idx(_idx - _n_frame, _n_frame)
    else:
        return _idx

def get_temporal_data(_idx, _flow_path, _n_frame, _in_channel=10):
    # video_path = os.path.join(self.root_dir, keys.split('-')[0])

    transform = set_transforms("temporal")
    result_data = []

    for i in _idx:
        flow = torch.FloatTensor(2 * _in_channel, 224, 224)
        for j in range(_in_channel):
            idx = reset_idx(i + j, _n_frame)
            frame_idx = "%05d.jpg" % idx

            x_image = os.path.join(_flow_path, 'flow_x_' + frame_idx)
            y_image = os.path.join(_flow_path, 'flow_y_' + frame_idx)

            imgX = (Image.open(x_image))
            imgY = (Image.open(y_image))

            X = transform(imgX)
            Y = transform(imgY)

            flow[2 * (j - 1), :, :] = X
            flow[2 * (j - 1) + 1, :, :] = Y
            imgX.close()
            imgY.close()
        result_data.append(torch.unsqueeze(flow, 0))

    return result_data

def get_spatial_data(_idx, _frame_path, _n_frame, _in_channel=1):
    # video_path = os.path.join(self.root_dir, keys.split('-')[0])

    transform = set_transforms('spatial')
    result_data = []

    for i in _idx:
        # cube = torch.FloatTensor(3, _in_channel, 224, 224)
        idx = reset_idx(i, _n_frame)
        # idx = i + j
        frame_idx = "image_%05d.jpg" % idx
        image = os.path.join(_frame_path, frame_idx)
        img = (Image.open(image))

        X = transform(img)
        # cube[:, j, :, :] = X
        img.close()
        result_data.append(torch.unsqueeze(X, 0))

    return result_data

def get_3dtsn_data(_idx, _frame_path, _n_frame, _in_channel=64):
    # video_path = os.path.join(self.root_dir, keys.split('-')[0])

    transform = set_transforms('3dtsn')
    result_data = []

    for i in _idx:
        cube = torch.FloatTensor(3, _in_channel, 224, 224)
        for j in range(_in_channel):
            idx = reset_idx(i + j, _n_frame)
            # idx = i + j
            frame_idx = "image_%05d.jpg" % idx
            image = os.path.join(_frame_path, frame_idx)
            img = (Image.open(image))

            X = transform(img)
            cube[:, j, :, :] = X
            img.close()
        result_data.append(torch.unsqueeze(cube, 0))

    return result_data

def get_input_data(extract_path, vid_name, mode):
    if mode == '3dtsn':
        data_root = os.path.join(extract_path, vid_name, "frames")
        n = len(os.listdir(data_root))
        idx = val_sample19(n, 0)
        return get_3dtsn_data(idx, data_root, n, 64)

    elif mode == 'spatial':
        data_root = os.path.join(extract_path, vid_name, "frames")
        n = len(os.listdir(data_root))
        idx = val_sample19(n, 0)
        return get_spatial_data(idx, data_root, n, 1)

    elif mode == 'temporal':
        data_root = os.path.join(extract_path, vid_name, "flow")
        n = len(os.listdir(data_root))
        idx = val_sample19(n / 2, 10)
        return get_temporal_data(idx, data_root, n / 2)



def get_model(root, mode):

    return torch.load(os.path.join(root, mode + ".pth"))

def one_hot_to_class(txt_path):
    result=dict()
    with open(txt_path, 'r') as f:
        for line in f.readlines():
            line = line.replace('\n', '')
            split_line = line.split(" ")

            result[str(split_line[-1])] = split_line[0].split('/')[0]
    f.close()

    with open('./label.pickle', 'wb') as f:
        pickle.dump(result, f)

    f.close()
    return result

def read_pickle(pickle_path):
    with open(pickle_path, 'rb') as f:
        result = pickle.load(f)

    return result

def forward(test_model, test_data):
    test_model.eval()
    for i, data in enumerate(test_data):
        if not i:
            pred = test_model(data.cuda()).data.cpu().numpy()
        else:
            pred = pred + test_model(data.cuda()).data.cpu().numpy()

        torch.cuda.empty_cache()
    return pred / len(test_data)

def visualize(data_root, pred_result):
    window_name = "Action Recognition Test"
    window = cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 600)

    for dat_name in sorted(os.listdir(data_root)):
        if not dat_name.split(".")[-1] == "jpg":
            continue
        frame = cv2.imread(os.path.join(data_root, dat_name))

        cv2.putText(frame, pred_result, (10, 30), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (255, 255, 255), 1)
        cv2.imshow(window_name, frame)
        cv2.waitKey(25)

def average_fusion(result_list, feature_ratio):
    n_feature = len(result_list)

    for i, pred in enumerate(result_list):
        tmp = None
        for j in range(n_feature):
            if tmp is None:
                tmp = pred * float(feature_ratio[j])
                continue
            tmp = tmp + pred * float(feature_ratio[j])

    return tmp

def main():
    global args
    args = parser.parse_args()

    # one-hot to text label table
    label_table = read_pickle("./label.pickle")
    models = args.model.split(" ")
    fusion_rate = args.fusion_rate.split(" ")

    print(args.model, "fusion model")
    video_name = extract_data(args.video, args.ext_path)
    print("Data extract complete")

    preds = list()
    for model in models:
        input_data = get_input_data(args.ext_path, video_name, mode=model)
        test_model = get_model(args.model_root, mode=model)
        preds.append(forward(test_model, input_data))
        torch.cuda.empty_cache()
        print(model, "predict complete")

    final_pred = average_fusion(preds, fusion_rate)
    pred_label = label_table[str(np.argmax(final_pred))]
    pred_label = pred_label.replace("_", " ")
    print("fusion complete")

    data_root = os.path.join(args.ext_path, video_name, "frames")
    visualize(data_root, pred_label)

if __name__ == '__main__':
    main()