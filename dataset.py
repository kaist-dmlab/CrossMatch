import numpy as np
import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import model
from tqdm import tqdm

import tensorflow as tf
# physical_devices = tf.config.list_physical_devices('GPU')
# try:
#     tf.config.experimental.set_memory_growth(physical_devices[0], True)
# except:
#     # Invalid device or cannot modify virtual devices once initialized.
#     pass
import numpy as np


def sample_first_timestamps(y, n, window_length, seed):
    '''
    # TODO: No matter what seed is, fix labeled timestamps w.r.t 'n' to reduce effect of initial labels.
    Args:
        x: input time series
        y: timestamp labels
        n: number of timestamps for each class

    Returns:
        mask vector for timestamp-wise classification. If mask=1, label exists and back propagation occurs at the timestamp.
    '''
    if n > 0:
        num_class = len(np.unique(y))
        mask = np.zeros_like(y)
        y_short = y[window_length // 2:-window_length // 2]
        np.random.seed(
            seed=0)  # fix sampled timestamp for each dataset. change 0 to seed when randomizing initial labels.
        for i in range(num_class):
            indice = np.where(y_short == i)[0]
            # print(indice,y,y_short)
            if len(indice) < n:
                print(f"number of timestamp label for class {i} is less than the number of required labels {n}")
            spld_indice = np.random.choice(indice, size=n, replace=False)
            spld_indice += window_length // 2
            mask[spld_indice] = 1
        np.random.seed()
    else:
        mask = np.ones_like(y)
        print("full labels for each class are used")
    return mask


def sample_true_timestamp_label_from_ratio(y, p, num_class):
    '''
    Return timestamp label array that p*len(y) timestamp labels are randomly changed to different labels.
    Args:
        y: true timestamp labels
        p: ratio of true timestamp labels
        num_class: the number of class

    Returns:
        Randomized y with ratio of p.
    '''
    if p == 1:
        return y
    NUM_SPL_TS = int(len(y) * p)
    np.random.seed(0)
    SPL_TS_TO_BE_CHGD = np.random.choice(list(range(len(y))), size=len(y)-NUM_SPL_TS, replace=False)
    np.random.seed()
    y[SPL_TS_TO_BE_CHGD] = np.random.choice(list(range(num_class)), size = len(SPL_TS_TO_BE_CHGD), replace=True)
    return y

def get_dataset(DATA):
    file_path = "datasets"
    X_long = np.load(os.path.join(file_path, DATA + "_X_long.npy"))
    y_long = np.load(os.path.join(file_path, DATA + "_y_long.npy"))
    y_seg_long = np.load(os.path.join(file_path, DATA + "_y_seg_long.npy"))
    file_boundaries = np.load(os.path.join(file_path, DATA + "_file_boundaries.npy"))
    print(f"{DATA} loaded from preprocessed files from {file_path}")
    print(X_long.shape, y_long.shape, y_seg_long.shape)
    return X_long, y_long, y_seg_long, file_boundaries


def train_test_generator(X, y, y_seg, mask, file_boundaries, seed, K):
    assert (seed <= K - 1)
    test_data_start = len(X) // K * seed
    if seed == K - 1:
        test_data_end = len(X)
    else:
        test_data_end = len(X) // K * (seed + 1)

    X_long_train = np.concatenate([X[:test_data_start], X[test_data_end:]])
    y_long_train = np.concatenate([y[:test_data_start], y[test_data_end:]])
    y_seg_long_train = np.concatenate([y_seg[:test_data_start], y_seg[test_data_end:]])
    mask_long_train = np.concatenate([mask[:test_data_start], mask[test_data_end:]])
    file_boundaries_train = np.concatenate(
        [file_boundaries[:test_data_start], file_boundaries[test_data_end:]])

    X_long_test = X[test_data_start:test_data_end]
    y_long_test = y[test_data_start:test_data_end]
    y_seg_long_test = y_seg[test_data_start:test_data_end]
    mask_long_test = mask[test_data_start:test_data_end]  # TODO: labeled_or_not index check needed
    file_boundaries_test = file_boundaries[test_data_start:test_data_end]
    print(f"NUM_CLASS: {len(np.unique(y))}")
    print(X_long_train.shape, y_long_train.shape, y_seg_long_train.shape, mask_long_train.shape,
          file_boundaries_train.shape)
    print(X_long_test.shape, y_long_test.shape, y_seg_long_test.shape, mask_long_test.shape,
          file_boundaries_test.shape)
    return X_long_train, y_long_train, y_seg_long_train, mask_long_train, file_boundaries_train, X_long_test, y_long_test, y_seg_long_test, mask_long_test, file_boundaries_test


def masked_timeseries_dataset(
    data,
    targets,
    mask,
    sequence_length,
    sequence_stride=1,
    sampling_rate=1,
    batch_size=128,
    shuffle=False,
    seed=None,
    start_index=None,
    end_index=None,
    center_timestamps=[]):
    if start_index:
        if start_index < 0:
            raise ValueError(f'`start_index` must be 0 or greater. Received: '
                             f'start_index={start_index}')
        if start_index >= len(data):
            raise ValueError(f'`start_index` must be lower than the length of the '
                             f'data. Received: start_index={start_index}, for data '
                             f'of length {len(data)}')
    if end_index:
        if start_index and end_index <= start_index:
            raise ValueError(f'`end_index` must be higher than `start_index`. '
                             f'Received: start_index={start_index}, and '
                             f'end_index={end_index} ')
        if end_index >= len(data):
            raise ValueError(f'`end_index` must be lower than the length of the '
                             f'data. Received: end_index={end_index}, for data of '
                             f'length {len(data)}')
        if end_index <= 0:
            raise ValueError('`end_index` must be higher than 0. '
                             f'Received: end_index={end_index}')

    # Validate strides
    if sampling_rate <= 0:
        raise ValueError(f'`sampling_rate` must be higher than 0. Received: '
                         f'sampling_rate={sampling_rate}')
    if sampling_rate >= len(data):
        raise ValueError(f'`sampling_rate` must be lower than the length of the '
                         f'data. Received: sampling_rate={sampling_rate}, for data '
                         f'of length {len(data)}')
    if sequence_stride <= 0:
        raise ValueError(f'`sequence_stride` must be higher than 0. Received: '
                         f'sequence_stride={sequence_stride}')
    if sequence_stride >= len(data):
        raise ValueError(f'`sequence_stride` must be lower than the length of the '
                         f'data. Received: sequence_stride={sequence_stride}, for '
                         f'data of length {len(data)}')

    if start_index is None:
        start_index = 0
    if end_index is None:
        end_index = len(data)

    # Determine the lowest dtype to store start positions (to lower memory usage).
    num_seqs = end_index - start_index - (sequence_length * sampling_rate) + 1
    if targets is not None:
        num_seqs = min(num_seqs, len(targets))
    if num_seqs < 2147483647:
        index_dtype = 'int32'
    else:
        index_dtype = 'int64'

    # Generate start positions
    # max_num_window = (end_index - start_index-sequence_length)//sequence_stride
    # for i in range(max_num_window):
    #     if np.sum(mask[i*sequence_stride:i*sequence_stride+sequence_length])>0 and (
    #     i*sequence_stride-sequence_length//2)>0:
    #         start_positions.append(i*sequence_stride-sequence_length//2)

    if len(center_timestamps)>0:
        start_positions = np.array(center_timestamps).astype(index_dtype)-sequence_length//2
    else:
        start_positions = np.array(np.where(mask == 1)[0] - sequence_length // 2, dtype=index_dtype)
    # print(len(start_positions), np.sort(start_positions))
    start_positions = start_positions[start_positions >= 0]
    # print(len(start_positions), np.sort(start_positions))
    # start_positions = np.arange(0, num_seqs, sequence_stride, dtype=index_dtype)
    if shuffle:
        if seed is None:
            seed = np.random.randint(1e6)
        rng = np.random.RandomState(seed)
        rng.shuffle(start_positions)

    sequence_length = tf.cast(sequence_length, dtype=index_dtype)
    sampling_rate = tf.cast(sampling_rate, dtype=index_dtype)

    positions_ds = tf.data.Dataset.from_tensors(start_positions).repeat()
    # print(start_positions, sequence_length, sampling_rate)
    # For each initial window position, generates indices of the window elements
    indices = tf.data.Dataset.zip(
        (tf.data.Dataset.range(len(start_positions)), positions_ds)).map(
        lambda i, positions: tf.range(  # pylint: disable=g-long-lambda
            positions[i],
            positions[i] + sequence_length * sampling_rate, #TODO: add varying sequence length
            sampling_rate),
        num_parallel_calls=tf.data.AUTOTUNE)

    dataset = sequences_from_indices(data, indices, start_index, end_index)
    if targets is not None:
        indices = tf.data.Dataset.zip(
            (tf.data.Dataset.range(len(start_positions)), positions_ds)).map(
            lambda i, positions: positions[i],
            num_parallel_calls=tf.data.AUTOTUNE)
        target_ds = sequences_from_indices(
            targets, indices, start_index, end_index)
        dataset = tf.data.Dataset.zip((dataset, target_ds))
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    if batch_size is not None:
        if shuffle:
            # Shuffle locally at each iteration
            dataset = dataset.shuffle(buffer_size=batch_size * 8, seed=seed)
        dataset = dataset.batch(batch_size)
    else:
        if shuffle:
            dataset = dataset.shuffle(buffer_size=1024, seed=seed)
    return dataset


def sequences_from_indices(array, indices_ds, start_index, end_index):
    dataset = tf.data.Dataset.from_tensors(array[start_index : end_index])
    dataset = tf.data.Dataset.zip((dataset.repeat(), indices_ds)).map(
            lambda steps, inds: tf.gather(steps, inds),  # pylint: disable=unnecessary-lambda
            num_parallel_calls=tf.data.AUTOTUNE)
    return dataset


def varying_context_timeseries_dataset(
    data,
    targets,
    overlap_length,
    sequence_length, # maximum window length = overlap_length + 2 * context_length
    sequence_stride=1,
    sampling_rate=1,
    batch_size=128,
    iterations=50000,
    varying_context_per_batch=True,
    shuffle=False,
    seed=None,
    start_index=None,
    end_index=None):
    '''
    Make batched windows with different length for each batch.

    :return: Tensorflow dataset
    '''
    if start_index:
        if start_index < 0:
            raise ValueError(f'`start_index` must be 0 or greater. Received: '
                             f'start_index={start_index}')
        if start_index >= len(data):
            raise ValueError(f'`start_index` must be lower than the length of the '
                             f'data. Received: start_index={start_index}, for data '
                             f'of length {len(data)}')
    if end_index:
        if start_index and end_index <= start_index:
            raise ValueError(f'`end_index` must be higher than `start_index`. '
                             f'Received: start_index={start_index}, and '
                             f'end_index={end_index} ')
        if end_index >= len(data):
            raise ValueError(f'`end_index` must be lower than the length of the '
                             f'data. Received: end_index={end_index}, for data of '
                             f'length {len(data)}')
        if end_index <= 0:
            raise ValueError('`end_index` must be higher than 0. '
                             f'Received: end_index={end_index}')

    # Validate strides
    if sampling_rate <= 0:
        raise ValueError(f'`sampling_rate` must be higher than 0. Received: '
                         f'sampling_rate={sampling_rate}')
    if sampling_rate >= len(data):
        raise ValueError(f'`sampling_rate` must be lower than the length of the '
                         f'data. Received: sampling_rate={sampling_rate}, for data '
                         f'of length {len(data)}')
    if sequence_stride <= 0:
        raise ValueError(f'`sequence_stride` must be higher than 0. Received: '
                         f'sequence_stride={sequence_stride}')
    if sequence_stride >= len(data):
        raise ValueError(f'`sequence_stride` must be lower than the length of the '
                         f'data. Received: sequence_stride={sequence_stride}, for '
                         f'data of length {len(data)}')

    if start_index is None:
        start_index = 0
    if end_index is None:
        end_index = len(data)

    # Determine the lowest dtype to store start positions (to lower memory usage).
    num_seqs = end_index - start_index - (sequence_length * sampling_rate) + 1
    if targets is not None:
        num_seqs = min(num_seqs, len(targets))
    if num_seqs < 2147483647:
        index_dtype = 'int32'
    else:
        index_dtype = 'int64'

    total_length = len(data)
    center_timestamps = np.arange(sequence_length,total_length-sequence_length,overlap_length)

    start_positions = np.array(center_timestamps).astype(index_dtype)-sequence_length//2
    start_positions = start_positions[start_positions >= 0].tolist()
    start_positions = start_positions*((iterations*batch_size)//len(start_positions))+start_positions[:(iterations*batch_size)%len(start_positions)]

    if shuffle:
        if seed is None:
            seed = np.random.randint(1e6)
        rng = np.random.RandomState(seed)
        rng.shuffle(start_positions)

    if varying_context_per_batch:
        context_lengths_sampled = np.repeat(np.random.choice(list(range(2,sequence_length-overlap_length,2)),size=iterations,replace=True), batch_size)
    else:
        context_lengths_sampled = np.repeat([sequence_length-overlap_length]*iterations, batch_size)

    context_lengths_sampled = tf.cast(context_lengths_sampled,dtype=index_dtype)
    context_lengths = tf.data.Dataset.from_tensors(context_lengths_sampled).repeat()

    sampling_rate = tf.cast(sampling_rate, dtype=index_dtype)

    positions_ds = tf.data.Dataset.from_tensors(start_positions).repeat()
    indices = tf.data.Dataset.zip(
        (tf.data.Dataset.range(len(start_positions)), positions_ds, context_lengths)).map(
        lambda i, positions, context_lengths: tf.range(  # pylint: disable=g-long-lambda
            positions[i],
            positions[i] + (overlap_length+context_lengths[i]) * sampling_rate, #TODO: add varying sequence length
            sampling_rate),
        num_parallel_calls=tf.data.AUTOTUNE)

    dataset = sequences_from_indices(data, indices, start_index, end_index)
    if targets is not None:
        indices = tf.data.Dataset.zip(
            (tf.data.Dataset.range(len(start_positions)), positions_ds)).map(
            lambda i, positions: positions[i],
            num_parallel_calls=tf.data.AUTOTUNE)
        target_ds = sequences_from_indices(
            targets, indices, start_index, end_index)
        dataset = tf.data.Dataset.zip((dataset, target_ds))
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    if batch_size is not None:
        if shuffle:
            # Shuffle locally at each iteration
            dataset = dataset.shuffle(buffer_size=batch_size * 8, seed=seed)
        dataset = dataset.batch(batch_size)
    else:
        if shuffle:
            dataset = dataset.shuffle(buffer_size=1024, seed=seed)
    return dataset



if __name__=="__main__":

    @tf.function
    def train_cls(model, x, y, lossMask, lambd=0.15):
        with tf.GradientTape() as tape:
            outputs = model.call_classifier(x, training=True)
            lossMask = tf.cast(lossMask, dtype=tf.float32)
            # cls_loss = model.cls_loss(y, outputs[0])
            # masked_cls_loss = tf.math.reduce_sum(tf.math.multiply(cls_loss,lossMask))
            # mean_masked_cls_loss =  tf.math.divide(masked_cls_loss,tf.math.reduce_sum(tf.cast(lossMask!=0,tf.float32)))
            # loss = mean_masked_cls_loss + lambd*model.seg_loss([],outputs[0])
            # print(f"{cls_loss}\n{masked_cls_loss}\n{mean_masked_cls_loss}")
            loss = tf.math.divide(tf.math.reduce_sum(tf.math.multiply(model.cls_loss(y, outputs[0]),lossMask)),tf.math.reduce_sum(tf.cast(lossMask!=0,tf.float32))) + lambd*model.seg_loss([],outputs[0])
            for i in range(len(model.tcn_stage)-1):
                loss += tf.math.divide(tf.math.reduce_sum(tf.math.multiply(model.cls_loss(y, outputs[i+1]),lossMask)), tf.math.reduce_sum(tf.cast(lossMask!=0,tf.float32))) + lambd*model.seg_loss([],outputs[i+1])
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    DATA = "50salads"
    file_path = "datasets"
    X_long = np.load(os.path.join(file_path, DATA + "_X_long.npy"))
    y_long = np.load(os.path.join(file_path, DATA + "_y_long.npy"))
    y_seg_long = np.load(os.path.join(file_path, DATA + "_y_seg_long.npy"))
    file_boundaries = np.load(os.path.join(file_path, DATA + "_file_boundaries.npy"))

    # mask = np.random.choice([0,1],size=len(y_long))
    mask = np.ones_like(y_long)
    mask = np.reshape(mask,(len(y_long),1))
    y_long = np.reshape(y_long,(len(y_long),1))
    X_mask_y = np.concatenate((X_long,mask,y_long),axis=1)
    NUM_CLASS = len(np.unique(y_long))
    dim = X_long.shape[1]

    print(X_mask_y.shape)

    X_mask_y_dataset = masked_timeseries_dataset(
        data=X_mask_y,
        targets=None,
        mask=mask,
        sequence_length=2048,
        sequence_stride=2048,
        shuffle=True,
        batch_size=1, )

    for example_inputs in X_mask_y_dataset.take(1):
        print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
    print(X_mask_y_dataset, len(X_mask_y_dataset))

    models = model.MSTCN(NUM_CLASS, lr=0.005, num_stage=1, num_filters=64)  # models include contrastive layer and classificaiton layer
    models(np.zeros((1, 2048, dim)))

    for i in range(50):
        for X_mask_y_batch in tqdm(X_mask_y_dataset, leave=False):
            X_batch = X_mask_y_batch[:,:,:-2]
            mask_batch = X_mask_y_batch[:,:,-2]
            y_batch = X_mask_y_batch[:,:,-1]
            loss = train_cls(models, X_batch, y_batch, mask_batch)
