import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import tensorflow as tf
import numpy as np
from dataset import masked_timeseries_dataset, varying_context_timeseries_dataset
from tensorflow.keras.utils import timeseries_dataset_from_array

def scaling(x, sigma=1.1):
    '''
    Apply same factor of scaling for each dimension in whole window batch
    :param X:
    :return:
    '''
    # https://arxiv.org/pdf/1706.00527.pdf
    factor = tf.random.normal(mean=2, stddev=sigma, shape=(x.shape[0], 1, x.shape[2]))
    factor = tf.tile(factor,[1,x.shape[1],1])
    factor = tf.cast(factor,tf.float64)
    return tf.math.multiply(x,factor)

def jittering(x, sigma=0.8):
    return x + tf.random.normal(mean=0, stddev=sigma, shape=x.shape, dtype=tf.float64)

class SingleWindow():
    '''
    Windowing and extract receptive area.
    '''
    def __init__(self, window_length, receptive_length):
        self.window_length = window_length
        self.receptive_length = receptive_length
        self.num_overlap = 1

    def sample_first_regions(self, y, label_length, n, seed):
        np.random.seed(seed=0)
        # np.random.seed(seed=seed)

        num_class = len(np.unique(y))
        mask = np.zeros_like(y)
        y_short = y[self.window_length // 2 + 1:-self.window_length // 2 - 1]

        # generate classwise first timestamp list for each segment where first timestamp + length does not cover boundary
        first_timestamp_list = []
        for i in range(num_class):
            y_short_class_ind = np.where(y_short == i)[0]
            boundaries = []
            segment_len = []
            length = 0
            for ind, (prev_ind, curr_ind) in enumerate(zip(y_short_class_ind, y_short_class_ind[1:])):
                length += 1
                if curr_ind != prev_ind + 1:
                    boundaries.append(curr_ind)
                    segment_len.append(length)
                    length = 0
            segment_len.append(length + 1)  # append last segment length
            boundaries.insert(0, y_short_class_ind[0])  # append first segment start position
            # make start ts from boundary start to boundary end (= start + length)
            boundaries = np.array(boundaries)

            ts_list = []
            # for ind_j, j in enumerate(boundaries):
            for ind_j, j in enumerate(boundaries[np.where(np.array(segment_len) > label_length)[0]]):
                ts_list += list(range(j, j + segment_len[ind_j] - label_length, label_length))
            # print(f"ts_list:{ts_list}")
            if len(ts_list) < n:
                print(f"Not enough segments, ts_list length:{len(ts_list)}, number of sampled seg:{n}")
            ts_list = np.array(ts_list) + self.window_length // 2
            spld_indice = np.random.choice(ts_list, size=n, replace=False).tolist()
            first_timestamp_list += spld_indice
        center_timestamps = []
        for ts in first_timestamp_list:
            mask[ts:ts + label_length] = 1
            center_timestamps.append(ts + label_length//2)

        np.random.seed()
        return mask, center_timestamps

    def sample_first_timestamp(self, y, n, seed):
        '''
        Args:
            x: input time series
            y: timestamp labels
            n: number of timestamps for each class

        Returns:
            mask vector for timestamp-wise classification. If mask=1, label exists and back propagation occurs at the
            timestamp.
        '''
        if n > 0:
            num_class = len(np.unique(y))
            mask = np.zeros_like(y)
            y_short = y[self.window_length // 2 + 1:-self.window_length // 2 - 1]
            np.random.seed(seed=0) # fix sampled timestamp for each dataset. change 0 to seed when randomizing initial labels.
            for i in range(num_class):
                indice = np.where(y_short==i)[0]
                # print(indice,y,y_short)
                if len(indice) < n:
                    print(f"number of timestamp label for class {i} is less than the number of required labels {n}")
                spld_indice = np.random.choice(indice,size=n,replace=False)
                spld_indice += self.window_length//2
                # print(spld_indice, y[spld_indice])
                mask[spld_indice]=1
            np.random.seed()
        else:
            mask = np.ones_like(y)
            print("full labels for each class are used")
        return mask


    def dataloader(self, X_mask_y, batch_size, mask=[], center_timestamps=[]):
        if len(mask) < 1:
            dataset = timeseries_dataset_from_array(data=X_mask_y, targets=None,
                                                    sequence_length=self.window_length,
                                                    sequence_stride=self.receptive_length, # self.overlap_length
                                                    start_index=self.receptive_length // 2,
                                                    end_index=X_mask_y.shape[0]-self.receptive_length // 2,
                                                    shuffle=True, batch_size=batch_size)
        else:
            dataset = masked_timeseries_dataset(data=X_mask_y, targets=None, mask=mask,
                                                sequence_length=self.window_length, sequence_stride=self.receptive_length,
                                                shuffle=True, batch_size=batch_size, center_timestamps=center_timestamps)
        return dataset


    def extract_overlap(self, output):
        start = self.window_length//2-self.receptive_length//2
        end = self.window_length//2+self.receptive_length//2
        output = tf.gather(output, tf.range(start, end), axis=1)

        return output

class Overlap():
    def __init__(self, window_length, overlap_length, start_position=None, end_position=None):
        self.window_length = window_length
        self.overlap_length = overlap_length
        if start_position == None:
            self.start_position = window_length-overlap_length
        else:
            self.start_position = start_position
        if end_position == None:
            self.end_position = window_length
        else:
            self.end_position = end_position
        self.total_length = self.window_length * 2 - self.overlap_length
        self.num_overlap = 2

    def sample_first_regions(self, y, label_length, n, seed):
        np.random.seed(seed=0)
        # np.random.seed(seed=seed)

        num_class = len(np.unique(y))
        mask = np.zeros_like(y)
        y_short = y[self.total_length // 2 + 1:-self.total_length]

        print(y.shape, np.unique(y))

        first_timestamp_list = []
        for i in range(num_class):
            y_short_class_ind = np.where(y_short == i)[0]
            boundaries = []
            segment_len = []
            length = 0
            for ind, (prev_ind, curr_ind) in enumerate(zip(y_short_class_ind, y_short_class_ind[1:])):
                length += 1
                if curr_ind != prev_ind + 1:
                    boundaries.append(curr_ind)
                    segment_len.append(length)
                    length = 0
            segment_len.append(length + 1)  # append last segment length
            boundaries.insert(0, y_short_class_ind[0])  # append first segment start position
            # make start ts from boundary start to boundary end (= start + length)
            boundaries = np.array(boundaries)

            ts_list = []
            for ind_j, j in enumerate(boundaries[np.where(np.array(segment_len) > label_length)[0]]):
                ts_list += list(range(j, j + segment_len[ind_j] - label_length, label_length))
            # print(f"ts_list:{ts_list}")
            if len(ts_list) < n:
                raise ValueError("Not enough segments")
            ts_list = np.array(ts_list) + self.total_length // 2
            spld_indice = np.random.choice(ts_list, size=n, replace=False).tolist()
            first_timestamp_list += spld_indice
        center_timestamps = []
        for ts in first_timestamp_list:
            mask[ts:ts + label_length] = 1
            center_timestamps.append(ts + label_length//2)

        np.random.seed()
        return mask, center_timestamps

    def sample_first_timestamp(self, y, n, seed):
        '''
        Args:
            x: input time series
            y: timestamp labels
            n: number of timestamps for each class

        Returns:
            mask vector for timestamp-wise classification. If mask=1, label exists and back propagation occurs at the
            timestamp.
        '''
        if n > 0:
            num_class = len(np.unique(y))
            mask = np.zeros_like(y)
            y_short = y[self.total_length // 2:-self.total_length // 2]
            np.random.seed(seed=0) # fix sampled timestamp for each dataset. change 0 to seed when randomizing initial labels.
            for i in range(num_class):
                indice = np.where(y_short==i)[0]
                # print(indice,y,y_short)
                if len(indice) < n:
                    print(f"number of timestamp label for class {i} is less than the number of required labels {n}")
                spld_indice = np.random.choice(indice,size=n,replace=False)
                spld_indice += self.total_length//2
                # print(spld_indice, y[spld_indice])
                mask[spld_indice]=1
            np.random.seed()
        else:
            mask = np.ones_like(y)
            print("full labels for each class are used")
        return mask


    def dataloader(self, X_mask_y, batch_size, num_iter, mask=[], center_timestamps=[]):
        if len(mask) < 1:
            dataset = varying_context_timeseries_dataset(data=X_mask_y,
                                                         targets=None,
                                                         overlap_length=self.overlap_length,
                                                         sequence_length=self.total_length, # maximum window length = overlap_length + 2 * context_length
                                                         batch_size=batch_size,
                                                         iterations=num_iter)
        else:
            dataset = masked_timeseries_dataset(data=X_mask_y, targets=None, mask=mask,
                                                sequence_length=self.total_length, sequence_stride=self.overlap_length,
                                                shuffle=True, batch_size=batch_size, center_timestamps=center_timestamps)
        return dataset


    def windowing(self, batch):
        # print(self.window_length, self.overlap_length, self.total_length)
        left = tf.gather(batch, tf.range(self.window_length), axis=1)
        right = tf.gather(batch, tf.range(self.window_length-self.overlap_length, self.total_length), axis=1)
        return tf.concat([left, right], axis=0)

    def extract_overlap(self, output):
        output_left, output_right = tf.split(output, 2, axis=0)
        output_left = tf.gather(output_left, tf.range(self.window_length-self.overlap_length, self.window_length), axis=1)
        output_right = tf.gather(output_right, tf.range(0,self.overlap_length), axis=1)
        output = tf.concat([output_left, output_right], axis=0)
        return output

    def extract_rest(self, output):
        output_left, output_right = tf.split(output, 2, axis=0)
        output_left = tf.gather(output_left, tf.range(0, self.window_length-self.overlap_length), axis=1)
        output_right = tf.gather(output_right, tf.range(self.overlap_length,self.window_length), axis=1)
        return output_left, output_right


if __name__ == "__main__":
    batch = 2
    dim = 3
    window_length = 8
    overlap_length = 2
    total_length = window_length*2-overlap_length
    data = tf.range(batch*total_length*dim)
    data = tf.reshape(data,(batch,total_length,dim))
    print(data)

    o = OverlapN(window_length,overlap_length)
    print(o.windowing(data))
    print(o.extract_overlap(o.windowing(data)))
