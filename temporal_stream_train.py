########################################
#     import requirement libraries     #
########################################
from keras.backend import set_session
from keras.optimizers import SGD
import os

# custom module
import data_loader
import network
from util import write_log, train_1epoch, validation_1epoch, save_best_model

# set the quantity of GPU memory consumed
import tensorflow as tf
config = tf.ConfigProto()
# use GPU memory in the available GPU memory capacity
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
set_session(sess)

# using pretrained model
pretrained_model_name = ''
using_pretrained_model = False
save_model_path = '/home/jm/workspace/Two-stream/flow_model'
num_epoch = 100
batch_size = 64

#########################################################
#                     CallBack  setup                   #
#########################################################
from keras.callbacks import TensorBoard, ModelCheckpoint
tbCallBack = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
# mcCallBack = ModelCheckpoint('./flow_result/{epoch:0}')


"""
pytorch code
def cross_modality_pretrain(conv1_weight, channel):
    # transform the original 3 channel weight to "channel" channel
    S=0
    for i in range(3):
        S += conv1_weight[:,i,:,:]
    avg = S/3.
    new_conv1_weight = torch.FloatTensor(64,channel,7,7)
    #print type(avg),type(new_conv1_weight)
    for i in range(channel):
        new_conv1_weight[:,i,:,:] = avg.data
    return new_conv1_weight

def weight_transform(model_dict, pretrain_dict, channel):
    weight_dict  = {k:v for k, v in pretrain_dict.items() if k in model_dict}
    #print pretrain_dict.keys()
    w3 = pretrain_dict['conv1.weight']
    #print type(w3)
    if channel == 3:
        wt = w3
    else:
        wt = cross_modality_pretrain(w3,channel)

    weight_dict['conv1_custom.weight'] = wt
    model_dict.update(weight_dict)
    return model_dict
"""



"""
class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))


history = AccuracyHistory()
"""
if __name__ == '__main__':

    #####################################################
    #     import requirement data using data loader     #
    #####################################################
    # HMDB-51 data loader
    root = '/home/jeongmin/workspace/data/HMDB51/preprocess/flow'
    train_txt_root = '/home/jm/Two-stream_data/HMDB51/train_split1.txt'
    test_txt_root = '/home/jm/Two-stream_data/HMDB51/test_split1.txt'

    train_loader = data_loader.DataLoader(root, batch_size=batch_size)
    train_loader.set_data_list(train_txt_root, train_test_type='train')

    train_val_loader = data_loader.DataLoader(root, batch_size=batch_size)
    train_val_loader.set_data_list(train_txt_root, train_test_type='test')

    test_loader = data_loader.DataLoader(root)
    test_loader.set_data_list(test_txt_root, train_test_type='test')

    print('complete setting data list')

    temporal = network.Temporal()
    #####################################################
    #     set convolution neural network structure      #
    #####################################################
    if using_pretrained_model:
        start_epoch_num = int(pretrained_model_name.split('_')[0]) + 1
        load_model_path = os.path.join(save_model_path, pretrained_model_name)
        temporal_stream = temporal.set_pretrained_model(load_model_path)

    else:
        start_epoch_num = 0
        temporal_stream = temporal.basic()
        print('set network')

    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    temporal_stream.compile(optimizer=sgd,  # 'Adam',
                            loss='categorical_crossentropy',
                            metrics=['accuracy'])
    print('complete network setting')

    tmp_numiter = len(train_loader.get_train_data_list())/batch_size
    num_iter = int(tmp_numiter)+1 if tmp_numiter - int(tmp_numiter) > 0 else int(tmp_numiter)
    tbCallBack.set_model(temporal_stream)

    best_val_acc = 0
    for epoch in range(start_epoch_num, start_epoch_num + num_epoch):
        print('Epoch', epoch)

        train_acc, train_loss = train_1epoch(temporal_stream, train_loader, num_iter)
        print("train_loss:", train_loss, "train_acc:", train_acc)

        tr_val_acc, tr_val_loss = validation_1epoch(temporal_stream, train_val_loader)
        print("tr_val_loss:", tr_val_loss, "tr_val_acc:", tr_val_acc)

        val_acc, val_loss = validation_1epoch(temporal_stream, test_loader)
        print("val_loss:", val_loss, "val_acc:", val_acc)

        write_log(tbCallBack,
                  ["train_loss", "train_acc", 'validation_loss', 'validation_acc', 'tr_val_loss', 'tr_val_acc'],
                  [train_loss, train_acc, val_loss, val_acc, tr_val_loss, tr_val_acc],
                  epoch)

        best_val_acc = save_best_model(epoch, val_acc, best_val_acc, temporal_stream, save_model_path)

    #  A  A
    # (‘ㅅ‘=)
    # J.M.Seo