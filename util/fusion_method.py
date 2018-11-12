import pickle
import numpy as np
import os
import re
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler

# experiments = ['3dtsn_resnet152', 'resnet152', 'resnet34', 'resnet101']
"""
experiments = ['3dtsn_resnet152_spatial',
               'resnet152_spatial', 'resnet152_temporal'#]#,
               'resnet34_spatial', 'resnet34_temporal'#],
               'resnet101_spatial', 'resnet101_temporal',
               'resnext101_spatial']
"""
experiments = ['resnext101_spatial', 'resnet101_temporal', 'resnet101_spatial']
# features = ['spatial', 'temporal']
file_attr = ['train', 'test']

text_root = "/home/jm/Two-stream_data/HMDB51"
pickle_root = "/home/jm/action_result"

def get_label(txt_root, split_num=1):

    label_dict = dict()
    for attr in file_attr:
        label_dict[attr] = dict()
        file_name = attr + "_split%d.txt" %split_num
        file_path = os.path.join(txt_root, file_name)

        with open(file_path, 'r') as f:
            for line in f.readlines():
                split_line = re.split("[ ]+", line)
                label_dict[attr][split_line[0]] = int(split_line[-1])

    return label_dict

def get_pickle(pickle_path):
    with open(pickle_path, 'rb') as f:
        output = pickle.load(f)
        first_key = list(output.keys())[0]
        first_data = output[first_key]

        """
        if np.max(first_data) >1 or np.min(first_data) <0:
            for i, key in enumerate(output):
                output[key] = softmax(output[key])
                # output[key] = sigmoid(output[key])
        """
    return output

# correct solution:
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference

def sigmoid(x, derivative=False):
    return x*(1-x) if derivative else 1/(1+np.exp(-x))

def average_fusion(result_list, true_label, feature_ratio):
    n_feature = len(result_list)
    n_video = len(result_list[0])
    n_true_vid = 0

    for i, key in enumerate(result_list[0]):
        tmp = None
        for j in range(n_feature):
            if tmp is None:
                tmp = result_list[j][key] * feature_ratio[j]
                continue
            tmp = tmp + result_list[j][key] * feature_ratio[j]

        if np.argmax(tmp) == true_label[key]:
            n_true_vid = n_true_vid + 1

    return n_true_vid/float(n_video)

def data_format_transform(result_list, true_label):
    n_feature = len(result_list)
    n_video = len(result_list[0])
    X = list()
    y = list()
    for i, key in enumerate(result_list[0]):
        tmp = None
        for j in range(n_feature):
            if tmp is None:
                tmp = result_list[j][key]
                continue
            tmp = np.concatenate((tmp, result_list[j][key]),axis=None)

        X.append(tmp)
        y.append(int(true_label[key]))

    return np.asarray(X), np.asarray(y)

def classifier_fusion(x_train, y_train, x_test, y_test, option):
    if option == 'svm':
        tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]},
                            {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
        score = ['precision', 'recall']
        model = GridSearchCV(SVC(), tuned_parameters, cv=5)

    elif option == 'rf':
        model = RandomForestClassifier(n_estimators=100, random_state=0)

    elif option == 'ada':
        model = AdaBoostClassifier(SVC(),algorithm="SAMME", n_estimators=200)
    else:
        raise NotImplementedError

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print(accuracy_score(y_test, y_pred, normalize=True))


def data_loader(_experiments, _file_attr):
    data = {"train":[],
            "test":[]}
    for experiment in _experiments:
        # for feature in _features:
        for attr in _file_attr:
            pickle_name = experiment + "_video_preds_" + attr + ".pickle"

            if not pickle_name in os.listdir(pickle_root):
                print(pickle_root)
                print(pickle_name)
                continue

            print(pickle_root)
            print(pickle_name)
            pickle_path = os.path.join(pickle_root, pickle_name)
            data[attr].append(get_pickle(pickle_path))

    return data

if __name__=="__main__":

    label = get_label(text_root)


    output = data_loader(experiments, file_attr)
    test_output = output['test']
    # train_output = output['train']
    ratio = [2, 1.5, 1]
    print(average_fusion(test_output, label['test'], ratio))
    """
    min_max_scaler = MinMaxScaler()
    X_train, y_train = data_format_transform(train_output, label['train'])
    X_train = min_max_scaler.fit_transform(X_train)
    X_test, y_test = data_format_transform(test_output, label['test'])
    X_test = min_max_scaler.transform(X_test)
    classifier_fusion(X_train, y_train, X_test, y_test, option='svm')
    """
    """
    with open('/home/jm/hdd/action_final_result/resnext101_spatial/resnext101_spatial_test.pickle', 'rb') as f:
        o = pickle.load(f)
        print(sorted(list(o.keys())))

    print(sorted(list(label['test'].keys())))
    """