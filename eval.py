import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix

def total_accuracy(self, y_pred, y_true):
    return np.sum(np.argmax(y_pred, axis=1).flatten() == y_true) / len(y_true)

def classwise_accuracy(y_pred, y_true):
    '''
    Actually this is recall.
    :param y_pred:
    :param y_true:
    :return:
    '''
    num_class = len(np.unique(y_true))
    # print(num_class)
    cls_acc = np.zeros(num_class)
    # y_pred = np.argmax(y_pred, axis=1).flatten()
    for i in range(num_class):
        class_num = np.sum(y_true==i)
        if class_num==0:
            cls_acc[i] = 0
        else:
            cls_acc[i] = np.sum(y_pred[y_true==i]==i)/class_num
    return cls_acc.tolist()

def classwise_precision_and_recall(y_pred,y_true,num_class):
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(num_class)))
    with np.errstate(divide='ignore', invalid='ignore'):
        recall = matrix.diagonal() / matrix.sum(axis=1)
        precision = matrix.diagonal() / matrix.sum(axis=0)
    recall[np.isnan(recall)]=0
    precision[np.isnan(precision)]=0
    return recall, precision

def get_labels_start_end_time(frame_wise_labels, bg_class=[]):
    labels = []
    starts = []
    ends = []
    last_label = frame_wise_labels[0]
    if frame_wise_labels[0] not in bg_class:
        labels.append(frame_wise_labels[0])
        starts.append(0)
    for i in range(len(frame_wise_labels)):
        if frame_wise_labels[i] != last_label:
            if frame_wise_labels[i] not in bg_class:
                labels.append(frame_wise_labels[i])
                starts.append(i)
            if last_label not in bg_class:
                ends.append(i)
            last_label = frame_wise_labels[i]
    if last_label not in bg_class:
        ends.append(i + 1)
    return labels, starts, ends


def levenstein(p, y, norm=False):
    m_row = len(p)
    n_col = len(y)
    D = np.zeros([m_row+1, n_col+1], np.float)
    for i in range(m_row+1):
        D[i, 0] = i
    for i in range(n_col+1):
        D[0, i] = i

    for j in range(1, n_col+1):
        for i in range(1, m_row+1):
            if y[j-1] == p[i-1]:
                D[i, j] = D[i-1, j-1]
            else:
                D[i, j] = min(D[i-1, j] + 1,
                              D[i, j-1] + 1,
                              D[i-1, j-1] + 1)

    if norm:
        score = (1 - D[-1, -1]/max(m_row, n_col)) * 100
    else:
        score = D[-1, -1]

    return score


def edit_score(recognized, ground_truth, norm=True, bg_class=[]):
    P, _, _ = get_labels_start_end_time(recognized, bg_class)
    Y, _, _ = get_labels_start_end_time(ground_truth, bg_class)
    return levenstein(P, Y, norm)


def f_score(recognized, ground_truth, overlap, bg_class=[]):
    p_label, p_start, p_end = get_labels_start_end_time(recognized, bg_class)
    y_label, y_start, y_end = get_labels_start_end_time(ground_truth, bg_class)

    tp = 0
    fp = 0

    IoU_list = []
    hits = np.zeros(len(y_label))

    for j in range(len(p_label)):
        intersection = np.minimum(p_end[j], y_end) - np.maximum(p_start[j], y_start)
        union = np.maximum(p_end[j], y_end) - np.minimum(p_start[j], y_start)
        IoU = (1.0*intersection / union)*([p_label[j] == y_label[x] for x in range(len(y_label))])
        # Get the best scoring segment
        idx = np.array(IoU).argmax()
        IoU_list.append(IoU[idx])
        if IoU[idx] >= overlap and not hits[idx]:
            tp += 1
            hits[idx] = 1
        else:
            fp += 1
    fn = len(y_label) - sum(hits)
    return float(tp), float(fp), float(fn), np.mean(IoU_list)

def boundary_accuracy(y_pred,y_seg):
    '''
    Get boundary accuracy in true predicted positive, false predicted positive, # true boundary range,
    # missed true boundary range.
    Args:
        y_pred: array of predicted timestamp-wise boundary label (1 means boundary).
        y_seg: array of predicted timestamp-wise boundary label (a true boundary is propagated to be a range).

    Returns:
        true predicted positive, false predicted positive, # true boundary range, # missed true boundary range
    '''
    assert(len(y_pred)==len(y_seg))
    true_pred, false_pred, num_true, not_detected = 0, 0, 0, 0

    prev_y = y_pred[0]

    for ts, y in enumerate(y_pred[1:]):
        if prev_y != y:
            if y_seg[ts] == 1:
                true_pred+=1
            else:
                false_pred+=1
        prev_y = y

    true_boundary_range_ts = np.where(y_seg==1)[0]
    # print(f"true_boundary_range_ts{true_boundary_range_ts}")
    prev_ts = true_boundary_range_ts[0]
    true_boundary_range_list = []
    boundary_range = [prev_ts]
    for ind, ts in enumerate(true_boundary_range_ts[1:]):
        # print(ind,ts,boundary_range)
        if ts != prev_ts+1:
            true_boundary_range_list.append(boundary_range)
            boundary_range = [ts]
        else:
            boundary_range.append(ts)
        prev_ts = ts
    true_boundary_range_list.append(boundary_range)
    # print(f"true_boundary_range_list{true_boundary_range_list}")
    num_true = len(true_boundary_range_list)
    for boundary_range in true_boundary_range_list:
        if np.sum(y_pred[boundary_range]) == 0:
            not_detected+=1

    return [true_pred, false_pred, num_true, not_detected]

# def get_all_metrics(y_pred,y_true,y_seg,bg_class=[]):
def get_all_metrics(y_pred,y_true,bg_class=[]):
    cls_ts_acc = classwise_accuracy(y_pred,y_true)
    # bound_acc = boundary_accuracy(y_pred,y_seg)
    metrics = []

    overlap = [.1, .25, .5]
    tp, fp, fn = np.zeros(3), np.zeros(3), np.zeros(3)

    correct = np.sum(y_pred==y_true)
    for s in range(len(overlap)):
        tp1, fp1, fn1, mean_IoU = f_score(y_pred, y_true, overlap[s], bg_class)
        tp[s] += tp1
        fp[s] += fp1
        fn[s] += fn1

    Acc = 100*float(correct)/len(y_pred)
    metrics.append(Acc)
    edit = edit_score(y_pred, y_true, True, bg_class)
    metrics.append(edit)
    # print(f"Acc: {Acc} edit:{edit}", end=" ")
    for s in range(len(overlap)):
        precision = tp[s] / float(tp[s]+fp[s])
        recall = tp[s] / float(tp[s]+fn[s])

        f1 = 2.0 * (precision*recall) / (precision+recall)

        f1 = np.nan_to_num(f1)*100
        metrics.append(f1)
        # print('F1@%0.2f: %.4f' % (overlap[s], f1), end=" ")
    # metrics+=bound_acc
    metrics+=cls_ts_acc
    return metrics


def test_model(model, num_class, X_long_test, y_long_test, y_seg_long_test, file_boundaries_test, window = 0):
    if window > 0:
        file_boundary_ind = np.arange(start=0,stop=len(y_long_test),step=window).tolist()
    elif window == 0:
        file_boundary_ind = np.where(file_boundaries_test == 1)[0].tolist()
    else:
        file_boundary_ind = []

    start = 0  # test_data_start_ind
    if len(file_boundary_ind) > 0:
        if not len(file_boundaries_test)-1 in file_boundary_ind:
            file_boundary_ind.append(len(file_boundaries_test)-1)
        for i in file_boundary_ind:
            output_final_file = model(X_long_test[np.newaxis, start:i+1]).numpy()[0]
            if start == 0:
                output_final = output_final_file
            else:
                output_final = np.concatenate([output_final, output_final_file], axis=0)
            start = i+1
    else:
        output_final = model(X_long_test[np.newaxis, :, :]).numpy()
    output_final = output_final.reshape((-1, num_class))
    y_test_flatten = tf.reshape(y_long_test, [-1]).numpy()
    if len(output_final)!=len(y_test_flatten):
        print(y_long_test.shape, X_long_test.shape, len(file_boundaries_test), file_boundary_ind, len(output_final), len(y_test_flatten))
        print("shapes are different when testing")

    return get_all_metrics(np.argmax(output_final,axis=1), y_long_test, y_seg_long_test)


if __name__ == "__main__":
    pred = np.zeros(1000)
    true = np.zeros(1000)
    true[10:100] = 1
    true[105:125] = 1
    true[205:225] = 1
    true[305:335] = 1

    print(get_all_metrics(pred,true))


