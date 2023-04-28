'''
Weak-strong matching (no overlapping window)
'''

# tcn embedding / classification input length
import numpy as np
import math
np.set_printoptions(precision=4)
emb_dim = 64

num_class_dict = {"50salads": 19, "HAPT": 6, "GTEA": 11, "mHealth": 12, "opportunity": 17}
dim_dict = {"50salads": 2048, "HAPT": 6, "GTEA": 2048, "mHealth": 23, "opportunity": 113}
batch_dict_l = {"50salads": 1, "HAPT": 4, "GTEA": 1, "mHealth": 4, "opportunity": 4}
batch_dict_u = {"50salads": 2, "HAPT": 8, "GTEA": 2, "mHealth": 8, "opportunity": 8}
num_label_per_class_dict = {"50salads": 20, "HAPT": 5, "GTEA": 10, "mHealth": 5, "opportunity": 2}
one_second_interval_dict = {"50salads": 30, "HAPT": 50, "GTEA": 15, "mHealth": 50, "opportunity": 100}
length_dict = {"50salads": 10, "HAPT": 10, "GTEA": 5, "mHealth": 10, "opportunity": 3}

lr_dict = {"50salads": 0.005, "HAPT": 0.005, "GTEA": 0.0005, "mHealth": 0.005, "opportunity": 0.005}
warmup_iter_dict = {"50salads": 25000, "HAPT": 5000, "GTEA": 10000, "mHealth": 15000, "opportunity": 17000}
iter_dict = {"50salads": 50000, "HAPT": 25000, "GTEA": 25000, "mHealth": 50000, "opportunity": 30000}
lambda1_dict = {"50salads": 1, "HAPT": 1, "GTEA": 1, "mHealth": 1, "opportunity": 1}
window_dict = {"50salads": 1536, "HAPT": 1536, "GTEA": 1536, "mHealth": 1280, "opportunity": 1152}
overlap_dict = {"50salads": 1024, "HAPT": 1024, "GTEA": 1024, "mHealth": 768, "opportunity": 1024}
dilation_dict = {"50salads": 10, "HAPT": 10, "GTEA": 10, "mHealth": 10, "opportunity": 10}


import argparse

parser = argparse.ArgumentParser(description='parameters for TSAL')
parser.add_argument('--data', type=str, default='None', help='dataset name')
parser.add_argument('--gpu', type=str, default="0", help='gpu number')
parser.add_argument('--seed', type=int, default=0, help='experiment seed')
parser.add_argument('--aug', type=str, default="None", help='augmentation method for pseudo-label match')
parser.add_argument('--mul_label_per_class', type=float, default=2.0, help='multiplier of the number of timestamp label per class')
parser.add_argument('--overlap', type=int, default=-1, help='the ratio of unlabeled batch size to labeled batch size')
parser.add_argument('--window', type=int, default=-1, help='input window length')

parser.add_argument('--pltest', type=int, default=0, help='strong augmentation name')
parser.add_argument('--lambda1', type=float, default=-1, help='hyperparameter for PL update')
parser.add_argument('--lambda2', type=float, default=-1, help='hyperparameter for balancing PL')
parser.add_argument('--cond', type=int, default=-1, help='warmup iteration')

args = parser.parse_args()
DATA = args.data
GPU = args.gpu
SEED = args.seed
AUG = args.aug

if args.overlap == -1:
    OVERLAP = overlap_dict[DATA]
else:
    OVERLAP = args.overlap
if args.window == -1:
    WINDOW = window_dict[DATA]
else:
    WINDOW = args.window

if args.lambda1 == -1:
    LAMBDA1 = lambda1_dict[DATA]
else:
    LAMBDA1 = args.lambda1
LAMBDA2 = args.lambda2

PL_TEST = args.pltest

LABEL_LENGTH = length_dict[DATA]


MUL_LABEL_PER_CLASS = args.mul_label_per_class
NUM_LABEL_PER_CLASS = int(num_label_per_class_dict[DATA] * MUL_LABEL_PER_CLASS)
NUM_CLASS = num_class_dict[DATA]

ITER = args.cond
if ITER == -1:
    ITER = warmup_iter_dict[DATA]

print(f"{args}\nNUM_LABEL_PER_CLASS: {NUM_LABEL_PER_CLASS}\nNUM_EMB: {emb_dim}\n")


import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    pass

import model
from eval import *
from dataset import *
from utils import *

from tqdm import tqdm
from augmentation import *

@tf.function
def pl_entropy_loss(outputs, mask, NUM_CLASS):
    '''

    :param outputs: classifier probability with shape(batch, timestamp, num_class).
    :param mask: mask indicating label existence with shape(batch, timestamp), an output of C3PL.
    :param NUM_CLASS
    :return: class size entropy loss, 1-entropy(prediction*log_c*prediction) where prediction is filtered and averaged.
    '''
    dim = outputs.shape[-1]
    mask = tf.expand_dims(mask, axis=2)
    tiled_mask = tf.tile(mask, [1, 1, dim])
    tiled_mask = tf.cast(tiled_mask, dtype=tf.bool)
    outputs_flat = tf.boolean_mask(outputs, tiled_mask)
    if tf.cast(tf.reduce_sum(mask),dtype=tf.bool):
        # tf.print(outputs_flat.shape)
        masked_outputs = tf.reshape(outputs_flat,(-1, outputs.shape[-1]))
        averaged_outputs = tf.reduce_mean(masked_outputs, axis=0)
        loss = 1-(-tf.tensordot(averaged_outputs,tf.math.log(averaged_outputs)/tf.math.log(tf.cast(NUM_CLASS,dtype=tf.float32)), axes=1))
        return loss
    else:
        return tf.cast(0, dtype=tf.float32)

# @tf.function
def masked_TMSE_loss(y_pred, mask_ind, multiples, max_value=4, reduction="mean"):
    '''

    :param y_pred: tensorflow tensor predicted from sequential classifier. shape=(batch,timestamp,dim)
    :return: return T-MSE loss for minimizing over-segmentation error.
    '''
    y_pred = tf.clip_by_value(y_pred, clip_value_min=1e-8, clip_value_max=1)

    one_timestamp = tf.constant([[0, 1]], dtype=tf.int32)
    multiples = tf.constant((mask_ind.shape[0], 1))
    one_timestamp = tf.tile(one_timestamp, multiples)
    prev_ind = tf.nn.relu(mask_ind - one_timestamp)
    prev_ind = tf.cast(prev_ind,dtype=tf.int32)
    prev_pred = tf.gather_nd(y_pred, prev_ind)
    curr_pred = tf.gather_nd(y_pred, mask_ind)

    delta_tc_square = tf.keras.metrics.mean_squared_error(tf.math.log(curr_pred),tf.stop_gradient(tf.math.log(prev_pred)))
    delta_tc_tilda = tf.clip_by_value(delta_tc_square, clip_value_min=0, clip_value_max=max_value**2)
    if reduction == "mean":
        return tf.math.reduce_mean(delta_tc_tilda)
    elif reduction == "none":
        return delta_tc_tilda
    else:
        raise NotImplementedError


@tf.function
def mstcn_loss(model, outputs, y, mask, lambd=0.15, epsilon=1e-6, add_TMSE_loss=True):
    '''

    Args:
        outputs: multi-stage output
        lambd: lambda for consistency loss
        epsilon: small number for preventing division error.
    Returns:
        mstcn_loss
    '''
    y = tf.cast(y, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    loss = tf.math.divide(tf.math.reduce_sum(tf.math.multiply(model.cls_loss(y, outputs[0]), mask)),
                          tf.math.reduce_sum(tf.cast(mask != 0, tf.float32)) + epsilon)
    if add_TMSE_loss:
        loss += lambd * model.seg_loss([], outputs[0])
        # loss += lambd * masked_TMSE_loss(outputs[0], mask = mask)
    for i in range(len(model.tcn_stage) - 1):
        loss += tf.math.divide(tf.math.reduce_sum(tf.math.multiply(model.cls_loss(y, outputs[i + 1]), mask)),
                               tf.math.reduce_sum(tf.cast(mask != 0, tf.float32)) + epsilon)
        if add_TMSE_loss:
            loss += lambd * model.seg_loss([], outputs[i + 1])
            # loss += lambd * masked_TMSE_loss(outputs[i + 1], mask = mask)

    return loss

@tf.function
def cross_entropy_with_soft_label(y, output_softmax):
    return tf.reduce_mean(-tf.reduce_sum(y * tf.math.log(output_softmax), [2]))

@tf.function
def mstcn_loss_soft_label(model, outputs, y, mask, lambd=0.15, epsilon=1e-6, add_TMSE_loss=True):
    '''

    Args:
        outputs: multi-stage output
        lambd: lambda for consistency loss
        epsilon: small number for preventing division error.
    Returns:
        mstcn_loss
    '''
    # total_loss = tf.reduce_mean(-tf.reduce_sum(y_true * tf.log(y_hat_softmax), [1]))
    y = tf.cast(y, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    loss = tf.math.divide(tf.math.reduce_sum(tf.math.multiply(cross_entropy_with_soft_label(y, outputs[0]), mask)),
                          tf.math.reduce_sum(tf.cast(mask != 0, tf.float32)) + epsilon)
    if add_TMSE_loss:
        loss += lambd * model.seg_loss([], outputs[0])
        # loss += lambd * masked_TMSE_loss(outputs[0], mask = mask)
    for i in range(len(model.tcn_stage) - 1):
        loss += tf.math.divide(tf.math.reduce_sum(tf.math.multiply(cross_entropy_with_soft_label(y, outputs[i + 1]), mask)),
                               tf.math.reduce_sum(tf.cast(mask != 0, tf.float32)) + epsilon)
        if add_TMSE_loss:
            loss += lambd * model.seg_loss([], outputs[i + 1])
            # loss += lambd * masked_TMSE_loss(outputs[i + 1], mask = mask)

    return loss

@tf.function
def classwise_averaged_loss(model, outputs, y, mask, lambd=0.15, epsilon=1e-6):
    # loss averaged over each class
    y = tf.cast(y, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)

    class_mask = tf.cast(y==0,dtype=tf.float32)
    class_true_mask = tf.math.multiply(mask,class_mask)
    masked_class_loss = tf.math.multiply(model.cls_loss(y, outputs[0]), class_true_mask)
    averaged_class_loss = tf.math.divide(tf.math.reduce_sum(masked_class_loss), tf.math.reduce_sum(tf.cast(class_true_mask != 0, tf.float32))+epsilon) + lambd * model.seg_loss([], outputs[0])
    for i in range(len(model.tcn_stage)-1):
        averaged_class_loss += tf.math.divide(tf.math.reduce_sum(tf.math.multiply(model.cls_loss(y, outputs[i+1]), class_true_mask)),
                                              tf.math.reduce_sum(tf.cast(class_true_mask != 0, tf.float32))+epsilon)\
                               + lambd * model.seg_loss([], outputs[i+1])
    for j in range(NUM_CLASS-1):
        class_mask = tf.cast(y==j,dtype=tf.float32)
        class_true_mask = tf.math.multiply(mask,class_mask)
        masked_class_loss = tf.math.multiply(model.cls_loss(y, outputs[0]), class_true_mask)
        averaged_class_loss += tf.math.divide(tf.math.reduce_sum(masked_class_loss), tf.math.reduce_sum(tf.cast(class_true_mask != 0, tf.float32))+epsilon) + lambd * model.seg_loss([], outputs[0])
        for i in range(len(model.tcn_stage)-1):
            averaged_class_loss += tf.math.divide(tf.math.reduce_sum(tf.math.multiply(model.cls_loss(y, outputs[i+1]), class_true_mask)),
                                                  tf.math.reduce_sum(tf.cast(class_true_mask != 0, tf.float32))+epsilon)\
                                   + lambd * model.seg_loss([], outputs[i+1])
    averaged_class_loss /= NUM_CLASS
    return averaged_class_loss


# @tf.function
def merge_pseudo_true_label(y_true, mask_true, pseudo_labels, pseudo_mask, ignore=False):
    '''
    Merge true label and pseudo-label whose prediction is above confidence.
    :param ignore:
    :param pseudo_mask:
    :param pseudo_labels:
    :param y_true: true class label with shape(batch, timestamp, 1)
    :param mask_true: mask indicating label existence with shape(batch, timestamp, 1)
    :return: binary merged mask (1 means pseudo or true label exist) and merged label.
    '''

    if ignore:
        return y_true, mask_true, -1, 1
    else:
        mask_true_inverted = tf.cast(tf.math.logical_not(tf.cast(mask_true, dtype=tf.bool)), dtype=tf.int32)
        pseudo_true_labels = tf.multiply(y_true, mask_true) + tf.multiply(pseudo_labels, mask_true_inverted)  # include every timestamp label from predictions and true label
        pseudo_true_mask = tf.multiply(pseudo_mask, mask_true_inverted) + mask_true

        pseudo_mask_bool = tf.cast(pseudo_true_mask, dtype=tf.bool)
        num_pseudo = tf.reduce_sum(pseudo_true_mask)
        num_corr_pseudo = tf.reduce_sum(tf.cast(tf.boolean_mask(y_true, pseudo_mask_bool) == tf.boolean_mask(pseudo_true_labels, pseudo_mask_bool),dtype=tf.int32))

        return pseudo_true_labels, pseudo_true_mask, num_pseudo, num_corr_pseudo

# @tf.function
def merge_soft_true_label(y_true, mask_true, soft_labels, pseudo_mask, ignore=False):
    if ignore:
        return y_true, mask_true, -1, 1
    else:
        mask_true = tf.cast(mask_true, dtype=tf.float32)
        pseudo_mask = tf.cast(pseudo_mask, dtype=tf.float32)
        soft_labels = tf.cast(soft_labels, dtype=tf.float32)
        mask_true_inverted = tf.cast(tf.math.logical_not(tf.cast(mask_true, dtype=tf.bool)), dtype=tf.float32)

        mask_true = tf.tile(tf.expand_dims(mask_true, axis=2),[1,1,NUM_CLASS])
        pseudo_mask = tf.tile(tf.expand_dims(pseudo_mask, axis=2),[1,1,NUM_CLASS])
        mask_true_inverted = tf.tile(tf.expand_dims(mask_true_inverted, axis=2),[1,1,NUM_CLASS])

        y_true_one_hot = tf.cast(tf.one_hot(y_true, depth=NUM_CLASS), dtype=tf.float32)
        pseudo_true_mask = tf.multiply(pseudo_mask, mask_true_inverted) + mask_true
        pseudo_true_labels = tf.multiply(y_true_one_hot, mask_true) + tf.multiply(soft_labels, mask_true_inverted)

        return pseudo_true_labels, tf.cast(pseudo_true_mask, dtype=tf.int32)

@tf.function
def normalize_one_hot_sum(one_hot_pl_sum):
    '''
    If a timestamp has single PL, then its weight becomes 1. Otherwise, weight sum of each PL at a timestamp becomes 1.
    :param one_hot_pl_sum: shape(batch,timestamp,num_class)
    :return: softened label
    '''
    sum_along_class_axis = tf.reduce_sum(one_hot_pl_sum, axis=2)
    sum_along_class_axis = tf.expand_dims(sum_along_class_axis,axis=2)
    soft_pl = one_hot_pl_sum/tf.tile(sum_along_class_axis,[1,1,NUM_CLASS])
    return soft_pl


@tf.function
def TimestampPL(model, aug_manager, X_l, mask_l, y_l, X_u, mask_u, y_u, sigma_t_c, PL_TEST, threshold=0.95, lambd1=1.0, temperature=1, iter=0, total_iter=25000):

    aug_weak_l = jittering(X_l)
    aug_weak_u = jittering(X_u)
    aug_strong_l = jittering(scaling(X_l))
    aug_strong_u = jittering(scaling(X_u))

    with tf.GradientTape() as tape:
        outputs_w_l = model.call_logit(aug_weak_l, training=True, temp=temperature)
        outputs_w_u = model.call_logit(aug_weak_u, training=True, temp=temperature)
        outputs_s_u = model.call_logit(aug_strong_u, training=True, temp=temperature)

        outputs_l_stage = []
        outputs_u_stage = []

        for output_w_l, output_s_u in zip(outputs_w_l[:-1], outputs_s_u[:-1]):
            outputs_l_stage.append(aug_manager.extract_overlap(output_w_l))
            outputs_u_stage.append(aug_manager.extract_overlap(output_s_u))

        y_l = aug_manager.extract_overlap(y_l)
        y_l = tf.cast(y_l, dtype=tf.int32)
        mask_l = aug_manager.extract_overlap(mask_l)
        mask_l = tf.cast(mask_l, dtype=tf.int32)

        outputs_u = tf.nn.softmax(outputs_w_u[-1],axis=2)
        outputs_u = aug_manager.extract_overlap(outputs_u)
        y_u = aug_manager.extract_overlap(y_u)
        y_u = tf.cast(y_u, dtype=tf.int32)
        mask_u = aug_manager.extract_overlap(mask_u)
        mask_u = tf.cast(mask_u, dtype=tf.int32)

        pseudo_labels_u = tf.cast(tf.argmax(tf.stop_gradient(outputs_u), axis=2), dtype=tf.int32)
        mask_confidence = tf.cast(tf.reduce_max(tf.stop_gradient(outputs_u), axis=2) >= threshold, dtype=tf.int32)
        pseudo_labels_u, pseudo_mask_u, num_pseudo, num_corr_pseudo = merge_pseudo_true_label(y_u, mask_u, pseudo_labels_u, mask_confidence)

        loss_pl_balance = pl_entropy_loss(outputs_u, pseudo_mask_u, NUM_CLASS)
        loss_l = mstcn_loss(model, outputs_l_stage, y_l, mask_l)



        if PL_TEST == 0 and iter > ITER: # FixMatch
            loss_u = mstcn_loss(model, outputs_u_stage, pseudo_labels_u, pseudo_mask_u)
            loss = loss_l + lambd1 * loss_u

        elif PL_TEST == 1 and iter > ITER: # FlexMatch
            T_t_c = sigma_t_c/tf.reduce_max(sigma_t_c)*threshold # dynamic thresholds for each class
            T_t_c = tf.cast(T_t_c/(2-T_t_c), dtype=tf.float32) # non-linear mapping
            pseudo_labels_u = tf.cast(tf.argmax(tf.stop_gradient(outputs_u), axis=2), dtype=tf.int32)
            confidence_u = tf.cast(tf.reduce_max(tf.stop_gradient(outputs_u), axis=2), dtype=tf.float32)
            batch_timestamp_thresholds = tf.reshape(tf.gather(T_t_c, tf.reshape(pseudo_labels_u,[-1])), confidence_u.shape)
            mask_confidence = tf.cast(tf.greater_equal(confidence_u, batch_timestamp_thresholds), dtype=tf.int32)
            pseudo_labels_u, pseudo_mask_u, num_pseudo, num_corr_pseudo = merge_pseudo_true_label(y_u, mask_u, pseudo_labels_u, mask_confidence)
            sigma_t_c += tf.math.bincount(tf.boolean_mask(pseudo_labels_u, pseudo_mask_u), minlength=NUM_CLASS) # update classwise pl number

            loss_u = mstcn_loss(model, outputs_u_stage, pseudo_labels_u, pseudo_mask_u)
            loss = loss_l + lambd1 * loss_u


        elif PL_TEST == 2 and iter > ITER: # FixMatch + PropReg
            loss_u = mstcn_loss(model, outputs_u_stage, pseudo_labels_u, pseudo_mask_u)
            loss = loss_l + lambd1 * loss_u + loss_pl_balance


        else:
            loss_u = tf.constant(0, dtype=tf.float32)
            loss = loss_l

    gradients = tape.gradient(loss, model.trainable_variables)
    model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss, loss_l, loss_u, outputs_l_stage[-1], outputs_u, y_u, mask_u, pseudo_labels_u, pseudo_mask_u, mask_confidence, sigma_t_c

def true_class_prob(outputs_u, y_u):
    onehot_mask = tf.cast(tf.one_hot(y_u,depth=NUM_CLASS),dtype=tf.bool)
    true_class_prob_total = tf.boolean_mask(outputs_u,onehot_mask) # 1D vector comes out
    y_squeeze = tf.reshape(y_u,[-1])
    true_class_prob_list = []
    true_class_prob_average_list = []
    for i in range(NUM_CLASS):
        classwise_pred = tf.gather_nd(true_class_prob_total, tf.where(y_squeeze == i)).numpy()
        true_class_prob_list.append(classwise_pred)
        true_class_prob_average_list.append(tf.reduce_mean(classwise_pred).numpy())
    return true_class_prob_list, np.array(true_class_prob_average_list)

def iou_weight_generator(context, overlap, receptive):
    num_intersection = np.convolve(np.ones(context+overlap), np.ones(receptive), "same")
    num_intersection = num_intersection[context:].tolist()
    left = num_intersection.copy()
    num_intersection.reverse()
    right = num_intersection.copy()
    return tf.cast(left, dtype=tf.float32), tf.cast(right, dtype=tf.float32)

def dual_batch_timematch(window_length, overlap_length):
    X_long, y_long, y_seg_long, file_boundaries = get_dataset(DATA)
    mask = np.zeros_like(y_long)  # dummy mask
    NUM_CLASS = len(np.unique(y_long))

    X_long_train, y_long_train, y_seg_long_train, mask_long_train_dummy, file_boundaries_train, X_long_test, \
    y_long_test, y_seg_long_test, mask_long_test_dummy, file_boundaries_test = train_test_generator(
        X_long, y_long, y_seg_long, mask, file_boundaries, seed=SEED, K=5)
    dim = X_long_train.shape[1]

    # Model Definition
    models = model.MSTCN(NUM_CLASS, lr=lr_dict[DATA], num_dilation=dilation_dict[DATA], num_stage=4, num_filters=emb_dim, total_iter=iter_dict[DATA], warmup_iter=warmup_iter_dict[DATA])

    # model initialization
    models.call_classifier(np.zeros((1, WINDOW, dim)))

    aug_manager = SingleWindow(window_length, overlap_length)

    mask_long_train, center_timestamps = aug_manager.sample_first_regions(y_long_train, LABEL_LENGTH, NUM_LABEL_PER_CLASS, SEED)
    print(sorted(center_timestamps))
    print(f"labeled timestamps: {np.sum(mask_long_train)}, ideal#: {(NUM_CLASS*LABEL_LENGTH*NUM_LABEL_PER_CLASS)},", f"timestamp label percentage{np.sum(mask_long_train)/len(mask_long_train)}")
    print(tf.unique_with_counts(tf.boolean_mask(y_long_train, mask_long_train)))

    y_train = np.reshape(y_long_train, (len(y_long_train), 1))
    mask_train = np.reshape(mask_long_train, (len(mask_long_train), 1))
    X_mask_y = np.concatenate((X_long_train, mask_train, y_train), axis=1)

    X_mask_y_dataset_labeled = aug_manager.dataloader(X_mask_y=X_mask_y, batch_size=batch_dict_l[DATA], mask=mask_train, center_timestamps=center_timestamps)
    X_mask_y_dataset_all = aug_manager.dataloader(X_mask_y=X_mask_y, batch_size=batch_dict_u[DATA])

    num_iter = iter_dict[DATA]
    num_measurement = iter_dict[DATA]//100

    X_mask_y_dataset_labeled = X_mask_y_dataset_labeled.repeat(int(num_iter / len(X_mask_y_dataset_labeled)) + 1).take(num_iter).shuffle(buffer_size=8*batch_dict_l[DATA])
    X_mask_y_dataset_all = X_mask_y_dataset_all.repeat(int(num_iter / len(X_mask_y_dataset_all)) + 1).take(num_iter).shuffle(buffer_size=8*batch_dict_u[DATA])
    X_mask_y_dataset_labeled_iter = iter(X_mask_y_dataset_labeled)
    X_mask_y_dataset_all_iter = iter(X_mask_y_dataset_all)


    i,j = 0,0
    results = []
    results_pl = []
    metric_l = []
    metric_u = []
    y_u_list = []

    y_u_batch_list = []
    pl_batch_list = []
    mask_batch_list = []

    pseudo_true_labels_u_list = []
    sum_num_corr_pseudo_l, sum_num_pseudo_l, sum_kl_l, sum_num_consistence_l, sum_num_corr_pseudo_u, sum_num_pseudo_u, sum_kl_u, sum_num_consistence_u, sum_num_corr_pseudo_u_conf, sum_num_pseudo_u_conf = 0,0,0,0,0,0,0,0,0,0
    sum_pl_per_cls = np.zeros(NUM_CLASS)
    pseudo_true_label_flatten_append = np.array([])
    true_label_flatten_append = np.array([])

    true_classwise_pred_prob_aver_list = []


    batch_bar = tqdm(range(num_iter), leave=False, ncols=200, position=0)
    sigma_t_c = tf.zeros(NUM_CLASS,dtype=tf.int32) # counting number of PL made at each iteration

    for ssl_iter in batch_bar:

        X_mask_y_batch_l = X_mask_y_dataset_labeled_iter.get_next()
        X_mask_y_batch_u = X_mask_y_dataset_all_iter.get_next()

        j+=1

        X_l = X_mask_y_batch_l[:, :, :-2]
        mask_l = X_mask_y_batch_l[:, :, -2]
        y_l = X_mask_y_batch_l[:, :, -1]

        X_u = X_mask_y_batch_u[:, :, :-2]
        mask_u = X_mask_y_batch_u[:, :, -2]
        y_u = X_mask_y_batch_u[:, :, -1]


        loss, loss_l, loss_u, outputs_l, outputs_u, y_u, mask_u, pseudo_labels_u, pseudo_mask_u, mask_confidence_u, sigma_t_c \
            = TimestampPL(models, aug_manager, X_l, mask_l, y_l, X_u, mask_u, y_u, sigma_t_c=sigma_t_c, temperature=1, PL_TEST=PL_TEST, lambd1=LAMBDA1, iter=tf.constant(j, dtype=tf.float32), total_iter=tf.constant(num_iter, dtype=tf.float32))


        pseudo_mask_bool = tf.cast(pseudo_mask_u, dtype=tf.bool)
        num_pseudo_u = tf.reduce_sum(pseudo_mask_u)
        num_corr_pseudo_u = tf.reduce_sum(tf.cast(tf.boolean_mask(y_u, pseudo_mask_bool) == tf.boolean_mask(pseudo_labels_u, pseudo_mask_bool),dtype=tf.int32))

        sum_loss_u = loss_u.numpy()
        sum_num_corr_pseudo_u += num_corr_pseudo_u.numpy()
        sum_num_pseudo_u += num_pseudo_u.numpy()


        pseudo_true_label_flatten = tf.reshape(tf.boolean_mask(pseudo_labels_u, pseudo_mask_u), [-1]).numpy() # masked, flattened pl
        true_label_flatten = tf.reshape(tf.boolean_mask(y_u,pseudo_mask_u), [-1]).numpy() # masked, flattened y
        pseudo_true_label_flatten_conf = tf.reshape(tf.boolean_mask(pseudo_labels_u, mask_confidence_u), [-1]).numpy()
        true_label_flatten_conf = tf.reshape(tf.boolean_mask(y_u, mask_confidence_u), [-1]).numpy()
        num_pl_per_cls = np.bincount(pseudo_true_label_flatten, minlength=NUM_CLASS)
        sum_pl_per_cls += num_pl_per_cls

        num_corr_pseudo_u_conf = np.sum(pseudo_true_label_flatten_conf==true_label_flatten_conf)
        num_pseudo_u_conf = np.sum(mask_confidence_u.numpy())
        with np.errstate(divide='ignore', invalid='ignore'):
            conf_pl_acc = num_corr_pseudo_u_conf/num_pseudo_u_conf
            if conf_pl_acc == np.nan:
                conf_pl_acc = 0
        sum_num_corr_pseudo_u_conf += num_corr_pseudo_u_conf
        sum_num_pseudo_u_conf += num_pseudo_u_conf

        pl_batch_list.append(pseudo_labels_u.numpy())
        mask_batch_list.append(mask_confidence_u.numpy())
        y_u_batch_list.append(y_u.numpy())

        y_u_list.append(true_label_flatten)
        pseudo_true_labels_u_list.append(pseudo_true_label_flatten)

        pseudo_true_label_flatten_append = np.append(pseudo_true_label_flatten_append, pseudo_true_label_flatten)
        true_label_flatten_append = np.append(true_label_flatten_append, true_label_flatten)

        true_classwise_pred_prob_list, true_classwise_pred_prob_aver = true_class_prob(outputs_u, y_u)
        true_classwise_pred_prob_aver_list.append(true_classwise_pred_prob_aver)

        batch_bar.set_description(f"fixmatch.py {DATA=} {WINDOW=} {OVERLAP=} {PL_TEST=} {SEED=} {GPU=} {loss_l=:.3f} {loss_u=:.3f} num_iter:{j}/{num_iter}")


        if (i == num_measurement or j==num_iter):
            print("\n")

            result = test_model(models, NUM_CLASS, X_long_test, y_long_test, y_seg_long_test, file_boundaries_test)
            results.append(result)

            with np.errstate(divide='ignore', invalid='ignore'):
                metric_u_sum = [sum_num_corr_pseudo_u/sum_num_pseudo_u, sum_num_corr_pseudo_u/num_measurement, sum_num_pseudo_u/num_measurement, (sum_num_corr_pseudo_u/num_measurement)/tf.size(pseudo_labels_u).numpy(), sum_num_consistence_u/num_measurement, sum_num_corr_pseudo_u_conf/sum_num_pseudo_u_conf, sum_num_corr_pseudo_u_conf/num_measurement, sum_num_pseudo_u_conf/num_measurement, sum_loss_u/num_measurement]
            metric_u.append(metric_u_sum)

            y_u_array = np.concatenate(y_u_list, axis=0)
            pseudo_true_labels_u_array = np.concatenate(pseudo_true_labels_u_list, axis=0)
            pl_precision, pl_recall = classwise_precision_and_recall(pseudo_true_labels_u_array, y_u_array, num_class=NUM_CLASS)
            pl_entropy = class_size_entropy(sum_pl_per_cls,NUM_CLASS)
            pl_metric = [pl_entropy]+pl_precision.tolist()+pl_recall.tolist()+sum_pl_per_cls.tolist()
            results_pl.append(pl_metric)

            print(f"\ntest result\n{result}\nmetric_u_sum\n{metric_u_sum}\npl_precision\n{pl_precision}\npl_entropy\n{pl_entropy}\nsum_pl_per_cls\n{sum_pl_per_cls}")
            print()
            y_u_list = []
            pseudo_true_labels_u_list = []
            i=0
            sum_num_corr_pseudo_l, sum_num_pseudo_l, sum_kl_l, sum_num_consistence_l, sum_num_corr_pseudo_u, sum_num_pseudo_u, sum_kl_u, sum_num_consistence_u, sum_num_corr_pseudo_u_conf, sum_num_pseudo_u_conf = 0,0,0,0,0,0,0,0,0,0
            sum_pl_per_cls = np.zeros(NUM_CLASS)
            pseudo_true_label_flatten_append = np.array([])
            true_label_flatten_append = np.array([])
        i += 1


    results = np.array(results)
    metric_l = np.array(metric_l)
    metric_u = np.array(metric_u)
    results_pl = np.array(results_pl)

    print(results)
    print(metric_l)
    print(metric_u)
    print(results_pl)


    LOG_FILE_NAME = f"FixMatch_{DATA}_{PL_TEST}_{MUL_LABEL_PER_CLASS}_{WINDOW}_{OVERLAP}_{LAMBDA1}_{ITER}_{SEED}"
    np.save(os.path.join(os.getcwd(), "metadata", f"Test_{LOG_FILE_NAME}.npy"), results)
    np.save(os.path.join(os.getcwd(), "metadata", f"metric_u_{LOG_FILE_NAME}.npy"), metric_u)
    np.save(os.path.join(os.getcwd(), "metadata", f"results_pl_{LOG_FILE_NAME}.npy"), results_pl)

    print(f"MAX TEST PERFORMANCE: {np.max(results[:, 0])}, {np.max(results[:, 1])}, {np.max(results[:, 2])}")


if __name__=="__main__":
    dual_batch_timematch(WINDOW, OVERLAP)
