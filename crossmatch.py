'''
varying-context windowing + Cross window matching
'''

# tcn embedding / classification input length
import numpy as np
np.set_printoptions(precision=4)
emb_length = 64
# input_length = 2048

# Fixed Hyperparams for classfication loss
# Tuned for 20 labels per class
num_class_dict = {"50salads": 19, "HAPT": 6, "GTEA": 11, "mHealth": 12, "opportunity": 17}
dim_dict = {"50salads": 2048, "HAPT": 6, "GTEA": 2048, "mHealth": 23, "opportunity": 113}
# batch_dict_l = {"50salads": 2, "HAPT": 8, "GTEA": 1, "mHealth": 8}
batch_dict_l = {"50salads": 1, "HAPT": 4, "GTEA": 1, "mHealth": 4, "opportunity": 4}
batch_dict_u = {"50salads": 2, "HAPT": 8, "GTEA": 2, "mHealth": 8, "opportunity": 8}
epoch_cls_dict_ssl = {"50salads": 200, "HAPT": 75, "GTEA": 75, "mHealth": 150}
one_second_interval_dict = {"50salads": 30, "HAPT": 50, "GTEA": 15, "mHealth": 50, "opportunity": 100}
num_label_per_class_dict = {"50salads": 20, "HAPT": 5, "GTEA": 10, "mHealth": 5, "opportunity": 2} # 220829 roll back to 20 for 50salads
length_dict = {"50salads": 10, "HAPT": 10, "GTEA": 5, "mHealth": 10, "opportunity": 3}


lr_dict = {"50salads": 0.005, "HAPT": 0.005, "GTEA": 0.0005, "mHealth": 0.005, "opportunity": 0.005} # 220829 lr for 50salads/GTEA changed for preventing overfitting
cond_dict = {"50salads": 25000, "HAPT": 5000, "GTEA": 5000, "mHealth": 15000, "opportunity": 17000}
iter_dict = {"50salads": 50000, "HAPT": 25000, "GTEA": 25000, "mHealth": 50000, "opportunity": 30000}
lambda1_dict = {"50salads": 1, "HAPT": 1, "GTEA": 1, "mHealth": 1, "opportunity": 1}
window_dict = {"50salads": 1536, "HAPT": 1536, "GTEA": 1536, "mHealth": 1536, "opportunity": 1088}
overlap_dict = {"50salads": 1024, "HAPT": 1024, "GTEA": 1024, "mHealth": 1024, "opportunity": 1024}
# window_dict = {"50salads": 512, "HAPT": 1024, "GTEA": 256, "mHealth": 1024} # 220917 video dataset use shorter window to avoid overfitting
# overlap_dict = {"50salads": 384, "HAPT": 768, "GTEA": 192, "mHealth": 768} # 220917 video dataset use shorter overlap to avoid overfitting
# window_dict = {"50salads": 1024, "HAPT": 1024, "GTEA": 1024, "mHealth": 1024} # 220920 video dataset use shorter window to avoid overfitting
# overlap_dict = {"50salads": 768, "HAPT": 768, "GTEA": 768, "mHealth": 768} # 220920 video dataset use shorter overlap to avoid overfitting
thres_dict = {"50salads": 0.95, "HAPT": 0.95, "GTEA": 0.95, "mHealth": 0.95, "opportunity": 0.95}
dilation_dict = {"50salads": 10, "HAPT": 10, "GTEA": 10, "mHealth": 10, "opportunity": 10} # 220920
# dilation_dict = {"50salads": 9, "HAPT": 11, "GTEA": 8, "mHealth": 11} # 220917 video dataset use shallower network to avoid overfitting

import argparse

parser = argparse.ArgumentParser(description='parameters for TSAL')
parser.add_argument('--data', type=str, default='None', help='dataset name')
parser.add_argument('--gpu', type=str, default="0", help='gpu number')
parser.add_argument('--seed', type=int, default=0, help='experiment seed')
parser.add_argument('--aug', type=str, default="None", help='augmentation method for pseudo-label match')
parser.add_argument('--mul_label_per_class', type=float, default=2.0, help='multiplier of the number of timestamp label per class')
parser.add_argument('--overlap', type=int, default=-1, help='the ratio of unlabeled batch size to labeled batch size')
parser.add_argument('--window', type=int, default=-1, help='input window length')
parser.add_argument('--stride', type=int, default=1, help='stride for fully supervised windowing')
parser.add_argument('--pltest', type=int, default=0, help='strong augmentation name')
parser.add_argument('--lambda1', type=float, default=-1, help='hyperparameter for PL update')
parser.add_argument('--lambda2', type=float, default=-1, help='hyperparameter for balancing PL')
parser.add_argument('--cond', type=int, default=-1, help='True label length')

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
STRIDE = args.stride
# DILATION = args.num_dilation

# WEAK_AUG = args.weakaug
# STRONG_AUG = args.strongaug

if args.lambda1 == -1:
    LAMBDA1 = lambda1_dict[DATA]
else:
    LAMBDA1 = args.lambda1
LAMBDA2 = args.lambda2 # 0.5 1.0 1.5 2.0 2.5

PL_TEST = args.pltest

LABEL_LENGTH = length_dict[DATA]
# LABEL_LENGTH = args.length
# LABEL_LENGTH = one_second_interval_dict[DATA]

MUL_LABEL_PER_CLASS = args.mul_label_per_class
NUM_LABEL_PER_CLASS = int(num_label_per_class_dict[DATA] * MUL_LABEL_PER_CLASS)
NUM_CLASS = num_class_dict[DATA]

ITER = args.cond
if ITER == -1:
    ITER = cond_dict[DATA]

print(f"{args}\nNUM_LABEL_PER_CLASS: {NUM_LABEL_PER_CLASS}\nNUM_EMB: {emb_length}\n")


import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
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
    y = tf.cast(y, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    loss = tf.math.divide(tf.math.reduce_sum(tf.math.multiply(cross_entropy_with_soft_label(y, outputs[0]), mask)),
                          tf.math.reduce_sum(tf.cast(mask != 0, tf.float32)) + epsilon)
    if add_TMSE_loss:
        loss += lambd * model.seg_loss([], outputs[0])
    for i in range(len(model.tcn_stage) - 1):
        loss += tf.math.divide(tf.math.reduce_sum(tf.math.multiply(cross_entropy_with_soft_label(y, outputs[i + 1]), mask)),
                               tf.math.reduce_sum(tf.cast(mask != 0, tf.float32)) + epsilon)
        if add_TMSE_loss:
            loss += lambd * model.seg_loss([], outputs[i + 1])
    return loss

@tf.function
def classwise_averaged_loss(model, outputs, y, mask, lambd=0.15, epsilon=1e-6):
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


@tf.function
def merge_pseudo_true_label(y_true, mask_true, pseudo_labels, pseudo_mask, ignore=False):
    '''
    Merge true label and pseudo-label whose prediction is above confidence.
    :param outputs: classifier probability with shape(batch, timestamp, num_class). The length of timestamp would be overlapped length.
    :param y_true: true class label with shape(batch, timestamp, 1)
    :param mask_true: mask indicating label existence with shape(batch, timestamp, 1)
    :param confidence: value to decide given timestamp output is pseudo-labeled or not.
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


@tf.function(reduce_retracing=True)
def OverlapPL(model, aug_manager_l, aug_manager_u, X_l, mask_l, y_l, X_u, mask_u, y_u, left_weight, right_weight, threshold, CONTEXT, sigma_t_c, PL_TEST, lambd1=1.0, temperature=1, iter=0):

    aug_weak_l = X_l
    aug_weak_u = X_u

    with tf.GradientTape() as tape:
        outputs_w_l = model.call_logit(aug_weak_l, training=True, temp=temperature)
        outputs_w_u = model.call_logit(aug_weak_u, training=True, temp=temperature)

        outputs_l_stage = []
        outputs_u_stage_left = []
        outputs_u_stage_right = []
        outputs_u_stage_overlap_left = []
        outputs_u_stage_overlap_right = []
        outputs_u_stage_rest_left = []
        outputs_u_stage_rest_right = []
        for output_w_l, output_s_u in zip(outputs_w_l[:-1], outputs_w_u[:-1]):
        # for output_w_l, output_s_u in zip(outputs_w_l[:-1], outputs_s_u[:-1]):
            outputs_l_stage.append(aug_manager_l.extract_overlap(output_w_l))
            outputs_u_overlap_left, outputs_u_overlap_right = tf.split(aug_manager_u.extract_overlap(output_s_u),2,axis=0)
            outputs_u_stage_overlap_left.append(outputs_u_overlap_left)
            outputs_u_stage_overlap_right.append(outputs_u_overlap_right)

            output_u_left, output_u_right = tf.split(output_s_u, 2, axis=0)
            outputs_u_stage_left.append(output_u_left) # batch,window,num_class
            outputs_u_stage_right.append(output_u_right) # batch,window,num_class

            rest_left, rest_right = aug_manager_u.extract_rest(output_s_u)
            outputs_u_stage_rest_left.append(rest_left)
            outputs_u_stage_rest_right.append(rest_right)

        y_l = aug_manager_l.extract_overlap(y_l)
        y_l = tf.cast(y_l, dtype=tf.int32)
        mask_l = aug_manager_l.extract_overlap(mask_l)
        mask_l = tf.cast(mask_l, dtype=tf.int32)



        ############ for pltest == 1,2,3,4,(single, cross, rest, soft, )  ###########
        outputs_u_window = tf.nn.softmax(outputs_w_u[-1],axis=2)

        y_u_window = tf.cast(y_u, dtype=tf.int32)
        mask_u_window = tf.cast(mask_u, dtype=tf.int32)

        y_u_overlap, _ = tf.split(aug_manager_u.extract_overlap(y_u_window),2,axis=0) # as y_u at overlap left, overlap right are same, just use (batch, overlap), not (2*batch, overlap)
        mask_u_overlap, _ = tf.split(aug_manager_u.extract_overlap(mask_u_window),2,axis=0)


        pseudo_labels_u = tf.cast(tf.argmax(tf.stop_gradient(outputs_u_window), axis=2), dtype=tf.int32) # left right pl mixed along axis 0.
        mask_confidence = tf.cast(tf.reduce_max(tf.stop_gradient(outputs_u_window), axis=2) >= threshold, dtype=tf.int32)
        pseudo_labels_u, pseudo_mask_u, num_pseudo, num_corr_pseudo = merge_pseudo_true_label(y_u_window, mask_u_window, pseudo_labels_u, mask_confidence) # whether or not use true label in unlabeled batch
        pseudo_labels_u_overlap_left, pseudo_labels_u_overlap_right = tf.split(aug_manager_u.extract_overlap(pseudo_labels_u),2,axis=0)
        pseudo_mask_u_overlap_left, pseudo_mask_u_overlap_right = tf.split(aug_manager_u.extract_overlap(pseudo_mask_u),2,axis=0)
        pseudo_mask_u_overlap_union = tf.cast((pseudo_mask_u_overlap_left + pseudo_mask_u_overlap_right) > 1, dtype=tf.int32)
        pseudo_mask_u_overlap_intersection = pseudo_mask_u_overlap_left * pseudo_mask_u_overlap_right
        loss_l = mstcn_loss(model, outputs_l_stage, y_l, mask_l)

        if PL_TEST == 0 and iter > ITER:

            pseudo_labels_u_overlap_left_one_hot = tf.one_hot(pseudo_labels_u_overlap_left, NUM_CLASS)
            pseudo_labels_u_overlap_right_one_hot = tf.one_hot(pseudo_labels_u_overlap_right, NUM_CLASS)

            norm_weight_left = tf.divide(left_weight, left_weight + right_weight)
            norm_weight_right = tf.divide(right_weight, left_weight + right_weight)

            pseudo_labels_u_overlap_left_weight = tf.tile(tf.reshape(norm_weight_left, (1,len(left_weight),1)),(pseudo_labels_u_overlap_left_one_hot.shape[0], 1, pseudo_labels_u_overlap_left_one_hot.shape[2]))
            pseudo_labels_u_overlap_right_weight = tf.tile(tf.reshape(norm_weight_right, (1,len(left_weight),1)),(pseudo_labels_u_overlap_right_one_hot.shape[0], 1, pseudo_labels_u_overlap_right_one_hot.shape[2]))

            pseudo_labels_u_overlap_sum = pseudo_labels_u_overlap_left_weight*pseudo_labels_u_overlap_left_one_hot + pseudo_labels_u_overlap_right_weight*pseudo_labels_u_overlap_right_one_hot
            pseudo_labels_u_overlap_soft = normalize_one_hot_sum(pseudo_labels_u_overlap_sum)

            loss_u_soft_pl_to_right_strong = 0.5*mstcn_loss_soft_label(model, outputs_u_stage_overlap_right, pseudo_labels_u_overlap_soft, pseudo_mask_u_overlap_union) # code for applying all timestamps where pl exists
            loss_u_soft_pl_to_left_strong = 0.5*mstcn_loss_soft_label(model, outputs_u_stage_overlap_left, pseudo_labels_u_overlap_soft, pseudo_mask_u_overlap_union)

            loss_u = loss_u_soft_pl_to_right_strong + loss_u_soft_pl_to_left_strong
            loss = loss_l + lambd1*loss_u

        elif PL_TEST == 1 and iter > ITER: # uni-directional matching per timestamp, PL made from confident window becomes target of output of inconfident window
            # assume convex locational confidence curve so that left part of overlap gets PL from left context window and vice versa.
            # ---------------|---------------
            #   leftPLused   |   rightPLused

            outputs_u_stage_right = []
            for  outputs_u_stage_overlap in outputs_u_stage_overlap_right:
                outputs_u_stage_right.append(tf.gather(outputs_u_stage_overlap,tf.range(0,OVERLAP//2),axis=1))
            pseudo_labels_u_overlap_left = tf.gather(pseudo_labels_u_overlap_left,tf.range(0,OVERLAP//2),axis=1)
            pseudo_mask_u_overlap_left = tf.gather(pseudo_mask_u_overlap_left,tf.range(0,OVERLAP//2),axis=1)

            outputs_u_stage_left = []
            for  outputs_u_stage_overlap in outputs_u_stage_overlap_left:
                outputs_u_stage_left.append(tf.gather(outputs_u_stage_overlap,tf.range(OVERLAP//2,OVERLAP),axis=1))
            pseudo_labels_u_overlap_right = tf.gather(pseudo_labels_u_overlap_right,tf.range(OVERLAP//2,OVERLAP),axis=1)
            pseudo_mask_u_overlap_right = tf.gather(pseudo_mask_u_overlap_right,tf.range(OVERLAP//2,OVERLAP),axis=1)

            # tf.print(outputs_u_stage_left[0].shape, pseudo_labels_u_overlap_left.shape, pseudo_mask_u_overlap_left.shape, outputs_u_stage_left[0].shape, pseudo_labels_u_overlap_right.shape, pseudo_mask_u_overlap_right.shape)

            loss_u_left = mstcn_loss(model, outputs_u_stage_right, pseudo_labels_u_overlap_left, pseudo_mask_u_overlap_left)
            loss_u_right = mstcn_loss(model, outputs_u_stage_left, pseudo_labels_u_overlap_right, pseudo_mask_u_overlap_right)

            loss_u = loss_u_left + loss_u_right
            loss = loss_l + lambd1 * loss_u

            ########### for logging ##########
            pseudo_labels_u_overlap_left = tf.concat([pseudo_labels_u_overlap_left,pseudo_labels_u_overlap_right], axis=1)
            pseudo_mask_u_overlap_left = tf.concat([pseudo_mask_u_overlap_left,pseudo_mask_u_overlap_right], axis=1)
            pseudo_labels_u_overlap_right = pseudo_labels_u_overlap_left
            pseudo_mask_u_overlap_right = pseudo_mask_u_overlap_left


        elif PL_TEST == 2 and iter > ITER: # Context varying FlexMatch


            outputs_u_stage_right = []
            for outputs_u_stage_overlap in outputs_u_stage_overlap_right:
                outputs_u_stage_right.append(tf.gather(outputs_u_stage_overlap, tf.range(0, OVERLAP // 2), axis=1))

            outputs_u_stage_left = []
            for  outputs_u_stage_overlap in outputs_u_stage_overlap_left:
                outputs_u_stage_left.append(tf.gather(outputs_u_stage_overlap,tf.range(OVERLAP//2,OVERLAP),axis=1))

            outputs_u_overlap = tf.concat([outputs_u_stage_left[-1],outputs_u_stage_right[-1]], axis=1)

            T_t_c = tf.cast(sigma_t_c / tf.reduce_max(sigma_t_c),dtype=tf.float32) * threshold  # dynamic thresholds for each class
            T_t_c = tf.cast(T_t_c / (2 - T_t_c), dtype=tf.float32)  # non-linear mapping
            pseudo_labels_u_overlap = tf.cast(tf.argmax(tf.stop_gradient(outputs_u_overlap), axis=2), dtype=tf.int32)
            confidence_u_overlap = tf.cast(tf.reduce_max(tf.stop_gradient(outputs_u_overlap), axis=2), dtype=tf.float32)
            batch_timestamp_thresholds = tf.reshape(tf.gather(T_t_c, tf.reshape(pseudo_labels_u_overlap,[-1])), confidence_u_overlap.shape)
            mask_confidence_overlap = tf.cast(tf.greater_equal(confidence_u_overlap, batch_timestamp_thresholds), dtype=tf.int32)
            pseudo_labels_u_overlap, pseudo_mask_u_overlap, num_pseudo, num_corr_pseudo = merge_pseudo_true_label(y_u_overlap, mask_u_overlap, pseudo_labels_u_overlap, mask_confidence_overlap)
            sigma_t_c += tf.math.bincount(tf.boolean_mask(pseudo_labels_u_overlap, pseudo_mask_u_overlap), minlength=NUM_CLASS) # update classwise pl number

            pseudo_labels_u_overlap_left = tf.gather(pseudo_labels_u_overlap,tf.range(0,OVERLAP//2),axis=1)
            pseudo_mask_u_overlap_left = tf.gather(pseudo_mask_u_overlap,tf.range(0,OVERLAP//2),axis=1)

            pseudo_labels_u_overlap_right = tf.gather(pseudo_labels_u_overlap,tf.range(OVERLAP//2,OVERLAP),axis=1)
            pseudo_mask_u_overlap_right = tf.gather(pseudo_mask_u_overlap,tf.range(OVERLAP//2,OVERLAP),axis=1)

            loss_u_left = mstcn_loss(model, outputs_u_stage_right, pseudo_labels_u_overlap_left, pseudo_mask_u_overlap_left)
            loss_u_right = mstcn_loss(model, outputs_u_stage_left, pseudo_labels_u_overlap_right, pseudo_mask_u_overlap_right)

            loss_u = loss_u_left + loss_u_right
            loss = loss_l + lambd1 * loss_u

            ########### for logging ##########
            pseudo_labels_u_overlap_left = tf.concat([pseudo_labels_u_overlap_left,pseudo_labels_u_overlap_right], axis=1)
            pseudo_mask_u_overlap_left = tf.concat([pseudo_mask_u_overlap_left,pseudo_mask_u_overlap_right], axis=1)
            pseudo_labels_u_overlap_right = pseudo_labels_u_overlap_left
            pseudo_mask_u_overlap_right = pseudo_mask_u_overlap_left

        elif PL_TEST == 3 and iter > ITER: # Context varying PropReg

            outputs_u_stage_right = []
            for  outputs_u_stage_overlap in outputs_u_stage_overlap_right:
                outputs_u_stage_right.append(tf.gather(outputs_u_stage_overlap,tf.range(0,OVERLAP//2),axis=1))
            pseudo_labels_u_overlap_left = tf.gather(pseudo_labels_u_overlap_left,tf.range(0,OVERLAP//2),axis=1)
            pseudo_mask_u_overlap_left = tf.gather(pseudo_mask_u_overlap_left,tf.range(0,OVERLAP//2),axis=1)

            outputs_u_stage_left = []
            for  outputs_u_stage_overlap in outputs_u_stage_overlap_left:
                outputs_u_stage_left.append(tf.gather(outputs_u_stage_overlap,tf.range(OVERLAP//2,OVERLAP),axis=1))
            pseudo_labels_u_overlap_right = tf.gather(pseudo_labels_u_overlap_right,tf.range(OVERLAP//2,OVERLAP),axis=1)
            pseudo_mask_u_overlap_right = tf.gather(pseudo_mask_u_overlap_right,tf.range(OVERLAP//2,OVERLAP),axis=1)

            loss_u_left = mstcn_loss(model, outputs_u_stage_right, pseudo_labels_u_overlap_left, pseudo_mask_u_overlap_left)
            loss_u_right = mstcn_loss(model, outputs_u_stage_left, pseudo_labels_u_overlap_right, pseudo_mask_u_overlap_right)

            outputs_merged = tf.concat([outputs_u_stage_left[-1],outputs_u_stage_right[-1]], axis=1)
            mask_merged = tf.concat([pseudo_mask_u_overlap_left,pseudo_mask_u_overlap_right], axis=1)

            PropRegLoss = pl_entropy_loss(outputs_merged, mask_merged, NUM_CLASS)

            loss_u = loss_u_left + loss_u_right
            loss = loss_l + lambd1 * loss_u + PropRegLoss

            ########### for logging ##########
            pseudo_labels_u_overlap_left = tf.concat([pseudo_labels_u_overlap_left,pseudo_labels_u_overlap_right], axis=1)
            pseudo_mask_u_overlap_left = mask_merged
            pseudo_labels_u_overlap_right = pseudo_labels_u_overlap_left
            pseudo_mask_u_overlap_right = pseudo_mask_u_overlap_left

        else:
            loss_l = mstcn_loss(model, outputs_l_stage, y_l, mask_l)
            loss_u = tf.constant(0, dtype=tf.float32)
            loss = loss_l

    gradients = tape.gradient(loss, model.trainable_variables)
    model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss, loss_l, loss_u, outputs_l_stage[-1], outputs_u_window, y_u_window, mask_u_window, pseudo_labels_u, pseudo_mask_u, mask_confidence, pseudo_labels_u_overlap_left, pseudo_mask_u_overlap_left, pseudo_labels_u_overlap_right, pseudo_mask_u_overlap_right, sigma_t_c

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

def balance_weight_generator(context, overlap):
    num_intersection_left = np.arange(context+overlap)+1
    num_intersection_right = num_intersection_left.tolist()
    num_intersection_right.reverse()
    num_intersection_right = np.array(num_intersection_right)
    num_intersection = np.stack([num_intersection_left, num_intersection_right])
    def entropy(x):
        x = x/np.sum(x)
        x_log = np.where(x != 0, np.log(x), 0)
        return -np.dot(x,x_log/np.log(2))
    entropy = np.apply_along_axis(func1d=entropy, arr=num_intersection, axis=0)

    num_intersection = entropy[context:].tolist()
    # print(entropy, num_intersection)
    left = num_intersection.copy()
    num_intersection.reverse()
    right = num_intersection.copy()
    return tf.cast(left, dtype=tf.float32), tf.cast(right, dtype=tf.float32)

def input_length_weight_generator(context, overlap, receptive):
    num_intersection = np.convolve(np.ones(context+overlap), np.ones(receptive), "same")
    num_intersection = num_intersection[context:].tolist()
    left = num_intersection.copy()
    num_intersection.reverse()
    right = num_intersection.copy()
    return tf.cast(left, dtype=tf.float32), tf.cast(right, dtype=tf.float32)

def reliability_function(x):
    return np.sqrt(1-(1-x)**2)

def weight_generator(context, overlap):
    # we assume left_context_length==right_context_length
    context = context
    left_window_left_context = np.arange(context,context+overlap)/overlap
    left_window_right_context = np.arange(overlap,0,-1)/overlap
    left_window_left_conf = reliability_function(left_window_left_context)
    left_window_right_conf = reliability_function(left_window_right_context)
    left = left_window_left_conf+left_window_right_conf
    right = np.flip(left)

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
    models = model.MSTCN(NUM_CLASS, lr=lr_dict[DATA], num_dilation=dilation_dict[DATA], num_stage=4, num_filters=emb_length, total_iter=iter_dict[DATA], warmup_iter=cond_dict[DATA])

    # model initialization
    models.call_classifier(np.zeros((1, WINDOW, dim)))

    aug_manager_l = Overlap(window_length, overlap_length)
    aug_manager_u = Overlap(window_length, overlap_length)

    mask_long_train, center_timestamps = aug_manager_l.sample_first_regions(y_long_train, LABEL_LENGTH, NUM_LABEL_PER_CLASS, SEED)
    print(sorted(center_timestamps))
    print(f"labeled timestamps: {np.sum(mask_long_train)}, ideal#: {(NUM_CLASS*LABEL_LENGTH*NUM_LABEL_PER_CLASS)},", f"timestamp label percentage{np.sum(mask_long_train)/len(mask_long_train)}")
    print(tf.unique_with_counts(tf.boolean_mask(y_long_train, mask_long_train)))

    y_train = np.reshape(y_long_train, (len(y_long_train), 1))
    mask_train = np.reshape(mask_long_train, (len(mask_long_train), 1))
    X_mask_y = np.concatenate((X_long_train, mask_train, y_train), axis=1)

    X_mask_y_dataset_labeled = aug_manager_l.dataloader(X_mask_y=X_mask_y, batch_size=batch_dict_l[DATA], mask=mask_train, center_timestamps=center_timestamps, num_iter=iter_dict[DATA])
    X_mask_y_dataset_all = aug_manager_u.dataloader(X_mask_y=X_mask_y, batch_size=batch_dict_u[DATA], num_iter=iter_dict[DATA])

    num_iter = iter_dict[DATA]
    num_measurement = iter_dict[DATA]//100

    X_mask_y_dataset_labeled = X_mask_y_dataset_labeled.repeat(int(num_iter / len(X_mask_y_dataset_labeled)) + 1).take(num_iter).shuffle(buffer_size=8*batch_dict_l[DATA])

    X_mask_y_dataset_labeled_iter = iter(X_mask_y_dataset_labeled)
    X_mask_y_dataset_all_iter = iter(X_mask_y_dataset_all)

    i,j = 0,0
    results = []
    metric_u = []
    results_pl = []
    y_u_list = []

    pseudo_true_labels_u_list = []
    sum_num_corr_pseudo_l, sum_num_pseudo_l, sum_kl_l, sum_num_consistence_l, sum_num_corr_pseudo_u, sum_num_pseudo_u, sum_kl_u, sum_num_consistence_u, sum_num_corr_pseudo_u_conf, sum_num_pseudo_u_conf = 0,0,0,0,0,0,0,0,0,0
    sum_pl_per_cls = np.zeros(NUM_CLASS)
    pseudo_true_label_flatten_append = np.array([])
    true_label_flatten_append = np.array([])


    batch_bar = tqdm(range(num_iter), leave=False, ncols=200, position=0)
    sigma_t_c = tf.zeros(NUM_CLASS, dtype=tf.int32)
    for ssl_iter in batch_bar:

        X_mask_y_batch_l = X_mask_y_dataset_labeled_iter.get_next()
        X_mask_y_batch_u = X_mask_y_dataset_all_iter.get_next()

        context_length = (X_mask_y_batch_u.shape[1]-overlap_length)//2
        aug_manager_u.window_length = context_length + overlap_length
        aug_manager_u.overlap_length = overlap_length
        aug_manager_u.total_length = aug_manager_u.window_length * 2 - aug_manager_u.overlap_length

        left_weight, right_weight = weight_generator(context_length, overlap_length)

        j+=1
        X_mask_y_batch_l = aug_manager_l.windowing(X_mask_y_batch_l)
        X_mask_y_batch_u = aug_manager_u.windowing(X_mask_y_batch_u)

        X_l = X_mask_y_batch_l[:, :, :-2]
        mask_l = X_mask_y_batch_l[:, :, -2]
        y_l = X_mask_y_batch_l[:, :, -1]

        X_u = X_mask_y_batch_u[:, :, :-2]
        mask_u = X_mask_y_batch_u[:, :, -2]
        y_u = X_mask_y_batch_u[:, :, -1]


        loss, loss_l, loss_u, outputs_l, outputs_u, y_u, mask_u_window, pseudo_labels_u, pseudo_mask_u, mask_confidence_u, pseudo_labels_u_overlap_left, pseudo_mask_u_overlap_left, pseudo_labels_u_overlap_right, pseudo_mask_u_overlap_right, sigma_t_c \
            = OverlapPL(models, aug_manager_l, aug_manager_u, X_l, mask_l, y_l, X_u, mask_u, y_u,  left_weight, right_weight, sigma_t_c=sigma_t_c, threshold=tf.cast(thres_dict[DATA], dtype=tf.float32), CONTEXT=tf.cast(context_length/(WINDOW-OVERLAP),tf.float32), temperature=1, PL_TEST=PL_TEST, lambd1=LAMBDA1, iter=tf.constant(j, dtype=tf.float32))

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
        # num_tl_per_cls = np.bincount(true_label_flatten, minlength=NUM_CLASS)
        num_tl_per_cls = np.bincount(y_u.numpy().flatten(), minlength=NUM_CLASS)
        # print(pseudo_true_label_flatten.shape, pseudo_true_labels_u.shape)
        sum_pl_per_cls += num_pl_per_cls

        num_corr_pseudo_u_conf = np.sum(pseudo_true_label_flatten_conf==true_label_flatten_conf)
        num_pseudo_u_conf = np.sum(mask_confidence_u.numpy())
        with np.errstate(divide='ignore', invalid='ignore'):
            conf_pl_acc = num_corr_pseudo_u_conf/num_pseudo_u_conf
            if conf_pl_acc == np.nan:
                conf_pl_acc = 0
        sum_num_corr_pseudo_u_conf += num_corr_pseudo_u_conf
        sum_num_pseudo_u_conf += num_pseudo_u_conf

        y_u_list.append(true_label_flatten)
        pseudo_true_labels_u_list.append(pseudo_true_label_flatten)

        pseudo_true_label_flatten_append = np.append(pseudo_true_label_flatten_append, pseudo_true_label_flatten)
        true_label_flatten_append = np.append(true_label_flatten_append, true_label_flatten)

        batch_bar.set_description(f"crossmatch.py {DATA=} {WINDOW=} {OVERLAP=} {PL_TEST=} {SEED=} {GPU=} {loss_l=:.3f} {loss_u=:.3f} num_iter:{j}/{num_iter}")

        if (i == num_measurement or j==num_iter):


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
            y_u_list = []
            pseudo_true_labels_u_list = []
            i=0
            sum_num_corr_pseudo_l, sum_num_pseudo_l, sum_kl_l, sum_num_consistence_l, sum_num_corr_pseudo_u, sum_num_pseudo_u, sum_kl_u, sum_num_consistence_u, sum_num_corr_pseudo_u_conf, sum_num_pseudo_u_conf = 0,0,0,0,0,0,0,0,0,0
            sum_pl_per_cls = np.zeros(NUM_CLASS)
            pseudo_true_label_flatten_append = np.array([])
            true_label_flatten_append = np.array([])
        i += 1


    results = np.array(results)
    metric_u = np.array(metric_u)
    results_pl = np.array(results_pl)

    print(results)
    print(metric_u)
    print(results_pl)


    LOG_FILE_NAME = f"CrossMatch_{DATA}_{PL_TEST}_{MUL_LABEL_PER_CLASS}_{WINDOW}_{OVERLAP}_{LAMBDA1}_{ITER}_{SEED}"
    np.save(os.path.join(os.getcwd(), "metadata", f"Test_{LOG_FILE_NAME}.npy"), results)
    np.save(os.path.join(os.getcwd(), "metadata", f"metric_u_{LOG_FILE_NAME}.npy"), metric_u)
    np.save(os.path.join(os.getcwd(), "metadata", f"results_pl_{LOG_FILE_NAME}.npy"), results_pl)
    print(f"MAX TEST PERFORMANCE: {np.max(results[:, 0])}, {np.max(results[:, 1])}, {np.max(results[:, 2])}")


if __name__=="__main__":

    # single_batch_timematch(WINDOW, DILATION)
    dual_batch_timematch(WINDOW, OVERLAP)
