import numpy as np
from sklearn.metrics import log_loss
import keras.backend as K
import keras.metrics as metrics
import tensorflow as tf
import progressbar
import os

def video_level_acc(_y_pred, _y_true):
    accuracy = metrics.categorical_accuracy(_y_true, _y_pred)
    return K.mean(accuracy, axis=0)


def video_level_loss(_y_pred, _y_true):
    _y_true = np.asarray(_y_true, dtype=np.float64)
    _y_pred = np.asarray(_y_pred, dtype=np.float64)

    return log_loss(_y_true, _y_pred)
"""
def video_level_loss(_y_pred, _y_true, _session):

    _y_pred = tf.convert_to_tensor(_y_pred, dtype='float64')
    _y_true = tf.convert_to_tensor(_y_true, dtype='float64')

    loss = metrics.categorical_crossentropy(_y_true, _y_pred)

    return loss
"""
def write_log(callback, names, logs, batch_no):
    for name, value in zip(names, logs):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        callback.writer.add_summary(summary, batch_no)
        callback.writer.flush()


def validation_1epoch(_model, _loader, _session):
    loss_list = []
    loss2_list = []
    correct = 0
    _loader.set_test_video_list()

    for i in progressbar.progressbar(range(len(_loader.get_test_data_list()))):
        _batch_x, _batch_y, eof = _loader.next_test_video()
        result = _model.predict_on_batch(_batch_x)

        label = np.mean(_batch_y, axis=0)
        predict = np.mean(result, axis=0)
        loss_list.append(video_level_loss(predict, label))
        # loss2_list.append(v_loss(predict, label, _session))

        if label.argmax() == predict.argmax():
            correct += 1

    return float(correct / len(loss_list)), np.asarray(loss_list).mean()


def train_1epoch(_model, _loader, _num_iter):
    # reset batch
    _loader.train_data_shuffle()
    loss_list = []
    acc_list = []
    for i in progressbar.progressbar(range(_num_iter)):
        _batch_x, _batch_y, _eof = _loader.next_train_batch()
        _batch_log = _model.train_on_batch(_batch_x, _batch_y)
        loss_list.append(_batch_log[0])
        acc_list.append(_batch_log[1])

        del _batch_x, _batch_y
        if _eof:
            break

    return np.mean(acc_list), np.mean(loss_list)


def save_best_model(_epoch, _val_acc, _best_val_acc, _model, _save_path):
    if not _epoch or _val_acc > _best_val_acc:
        model_name = os.path.join(_save_path, 'best_model.h5')
        _model.save(model_name)
        print("Save Best model to disk")

        return _val_acc
