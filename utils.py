import numpy as np
from matplotlib import pyplot as plt
import matplotlib.colors as pltc
import tensorflow as tf

def class_size_entropy(class_size_vector, num_class):
    with np.errstate(divide='ignore', invalid='ignore'):
        class_size_vector = class_size_vector/np.sum(class_size_vector)
        log_class_size = np.where(class_size_vector != 0, np.log(class_size_vector), 0)
        return -np.dot(class_size_vector,log_class_size/np.log(num_class))

def compute_segment_location(y, indice_start, indice_end):
    duration = []
    ind = 0
    for label_ts_prev, label_ts in zip(y[indice_start:indice_end - 1], y[indice_start + 1:indice_end]):
        ind += 1
        if label_ts_prev != label_ts:
            duration.append([int(y[ind - 1]), ind])
    if duration[-1][-1] < indice_end:
        duration.append([int(y[-1]), indice_end - indice_start])
    return duration

def graph(X, y, start, length, graph_ind):
    all_colors = [k for k, v in pltc.cnames.items()]

    indice_start = start
    indice_end = start+length

    duration = compute_segment_location(y, indice_start, indice_end)

    plt.figure(figsize=(15, 2))
    if len(X) != 0 and (X.shape[1] < 100):
        plt.plot(X[indice_start:indice_end])

    axv_ind = 0
    for label, ind in duration:
        if axv_ind == 0:
            plt.axvspan(0, ind, color=all_colors[label], alpha=0.5)
            plt.text(ind / 2, 0.5, str(label), ha='center')
        else:
            plt.axvspan(prev_ind, ind, color=all_colors[label], alpha=0.5)
            plt.text((ind + prev_ind) / 2, 0.5, str(label), ha='center')
        axv_ind += 1
        prev_ind = ind


    plt.xlabel("Timestamp")
    plt.ylabel("Raw value")
    plt.tight_layout()

    plt.savefig("./figures/" + str(graph_ind) + ".png", dpi=300, bbox_inches='tight')
    plt.close()

def search_segment_from_timestamp(y,t):
    '''

    Args:
        y: true timestamp label
        t: query timestamp

    Returns: timestamp list of the segment where containing t

    '''
    duration = compute_segment_location(y,0,len(y))
    seg_ind = []
    for dur_ind, dur_list in enumerate(duration): # duration is sorted from past timestamps to end timestamps
        # print(dur_ind,dur_list)
        if t <= dur_list[1]:
            seg_ind = dur_ind
            # seg_cls = dur_list[0]
            break
    if type(seg_ind)==list:
        print("Segment Not Found from:",len(y),t)
        assert(False)
    if seg_ind == 0:
        return list(range(0,duration[seg_ind][1]))
    else:
        return list(range(duration[seg_ind-1][1],duration[seg_ind][1]))


def sampler_testing(y, y_coh, seg_tree_true, seg_tree, name, batch_size, embs):

    if name =="test1":
        batch_indice = np.random.choice(list(range(len(y))), size=batch_size, replace=False).tolist()
        num_pos = 6
        num_neg = 250
        pos_ts = []
        neg_ts = []
        for i in range(batch_size):
            pts = np.random.choice(list(range(len(y))), size=num_pos, replace=False).tolist()
            pos_ts.append(pts)
            nts = np.random.choice(list(range(len(y))), size=num_neg, replace=False).tolist()
            neg_ts.append(nts)
    elif name == "test.txt":
        batch_indice = np.random.choice(list(range(len(y))), size=batch_size, replace=False).tolist()
        num_pos = 6
        num_neg = 250
        pos_mask = []
        neg_mask = []
        for i in range(batch_size):
            pts = np.random.choice(list(range(len(y))), size=num_pos, replace=False).tolist()
            pmask = np.zeros((len(y),embs.shape[2]))
            pmask[pts,:] = 1
            pos_mask.append(pmask)
            pts = np.random.choice(list(range(len(y))), size=num_neg, replace=False).tolist()
            nmask = np.zeros((len(y), embs.shape[2]))
            nmask[pts, :] = 1
            neg_mask.append(nmask)

    return tf.constant(batch_indice, dtype=tf.int32), tf.constant(pos_mask, dtype=tf.int32), tf.constant(neg_mask, dtype=tf.int32)

if __name__ == "__main__":
    size = 1000
    num_seg = 100
    y = np.zeros(size)
    bound_ind = np.random.choice(np.arange(size), size=num_seg, replace=False)
    bound_ind = sorted(bound_ind)
    prev_i = 0
    cls = 0
    for i in bound_ind:
        # print(prev_i,i)
        y[prev_i:i] = cls
        prev_i = i
        cls+=1
    # if not (len(y)-1 in bound_ind):
    #     bound_ind.append(len(y)-1)
    # print(y)
    ts_list = search_segment_from_timestamp(y,len(y)-1)
    print(ts_list)