import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv1D, BatchNormalization, ReLU, Dropout, Softmax, InputLayer
from tensorflow.keras import Model
from tensorflow.keras.losses import Loss
import numpy as np
import math


class TMSE_loss(Loss):
    def call(self, y_true, y_pred, max_value=4, reduction="mean"):
        '''

        :param y_pred: tensorflow tensor predicted from sequential classifier. shape=(batch,timestamp,dim)
        :return: return T-MSE loss for minimizing over-segmentation error.
        '''
        # # delta_tc = tf.clip_by_value(tf.math.abs(tf.math.log(tf.math.divide(y_pred[:,1:,:],y_pred[:,:-1,:]))), clip_value_min=0, clip_value_max=np.sqrt(max_value))
        # delta_tc = tf.math.abs(tf.math.log(y_pred[:,1:,:])-tf.stop_gradient((tf.math.log(y_pred[:,:-1,:]))))
        # delta_tc_square = tf.math.square(delta_tc)
        # delta_tc_tilda = tf.clip_by_value(delta_tc_square, clip_value_min=0, clip_value_max=max_value)
        # tmse = tf.math.reduce_mean(delta_tc_tilda)
        # print(tmse)
        y_pred = tf.clip_by_value(y_pred, clip_value_min=1e-8, clip_value_max=1)
        delta_tc_square = tf.keras.metrics.mean_squared_error(tf.math.log(y_pred[:,1:,:]),tf.stop_gradient(tf.math.log(y_pred[:,:-1,:])))
        delta_tc_tilda = tf.clip_by_value(delta_tc_square, clip_value_min=0, clip_value_max=max_value**2)
        if reduction == "mean":
            return tf.math.reduce_mean(delta_tc_tilda)
        elif reduction == "none":
            return delta_tc_tilda
        else:
            raise NotImplementedError

class FeatureNormalization(tf.keras.layers.Layer):
    def __init__(self):
        super(FeatureNormalization, self).__init__()

    def build(self, input_shape):
        return

    def call(self, inputs):
        return tf.math.l2_normalize(inputs, axis=2)

class CosineSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate, total_iter, warmup_iter):
        self.initial_learning_rate = tf.constant(initial_learning_rate, dtype=tf.float32)
        self.warmup_iter = warmup_iter
        self.total_iter = total_iter - self.warmup_iter


    def __call__(self, step):
        if step < self.warmup_iter:
            return self.initial_learning_rate
        else:
            step -= self.warmup_iter
            cosine_decay = self.initial_learning_rate * tf.math.cos(7 * tf.constant(math.pi) * step / (16 * self.total_iter))
            return cosine_decay



class MSTCN(Model):
    def __init__(self, num_class, lr=0.0005, num_stage=1, kernel_size=3, num_filters=64, num_dilation=11, dropout_rate=0.5, total_iter=25000, warmup_iter=4000,*args, **kwargs):
        super(MSTCN, self).__init__(*args, **kwargs)
        self.num_class = num_class
        self.num_dilation = num_dilation
        self.num_stage = num_stage
        self.num_filters = num_filters
        self.tcn_stage = []


        for j in range(num_stage):
            tcn = []
            tcn.append(Conv1D(filters=num_filters, kernel_size=1, strides=1, padding='same'))
            for i in range(num_dilation):
                dilated_conv = []
                dilated_conv.append(Conv1D(filters=num_filters, kernel_size=kernel_size, strides=1, padding='same', dilation_rate = [2 ** i]))
                dilated_conv.append(BatchNormalization())
                dilated_conv.append(ReLU())
                dilated_conv.append(Conv1D(filters=num_filters, kernel_size=1, strides=1, padding='same'))
                dilated_conv.append(Dropout(rate=dropout_rate))
                tcn.append(dilated_conv)
            tcn.append(FeatureNormalization())
            tcn.append(Conv1D(filters=num_class, kernel_size=1, strides=1, padding='same')) # timestamp classification layer
            tcn.append(Softmax())
            self.tcn_stage.append(tcn)
        self.proj_layers = [Conv1D(filters=num_filters, kernel_size=1, strides=1, padding='same',activation="relu"),FeatureNormalization()]



        self.cls_loss = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        self.seg_loss = TMSE_loss()
        self.seg_loss_no_reduction = TMSE_loss(reduction="none")
        self.cosine_lr_schedule = CosineSchedule(lr, total_iter, warmup_iter)

        self.optimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9, nesterov=True)

    def call(self, x, training = False): # generate softmax output

        for i in range(len(self.tcn_stage)):
            for j in range(self.num_dilation + 4):
                if 0 < j and j < self.num_dilation + 1:
                    input = x
                    for k in range(5):
                        x = self.tcn_stage[i][j][k](x, training=training)
                    x = input + x
                else:
                    x = self.tcn_stage[i][j](x, training=training)
        return x

    def call_classifier(self, x, training = False):

        outputs = []
        for i in range(len(self.tcn_stage)): # loop for
            for j in range(self.num_dilation + 4):
                if 0 < j and j < self.num_dilation + 1:
                    input = x
                    for k in range(5):
                        x = self.tcn_stage[i][j][k](x, training=training)
                    x = input + x
                else:
                    x = self.tcn_stage[i][j](x, training=training)
            outputs.append(x)
        return outputs


    def call_logit(self, x, training=False, temp=1):

        outputs = []
        for i in range(len(self.tcn_stage)):  # loop for
            for j in range(self.num_dilation + 4):
                if 0 < j and j < self.num_dilation + 1:
                    input = x
                    for k in range(5):
                        x = self.tcn_stage[i][j][k](x, training=training)
                    x = input + x
                else:
                    if i==len(self.tcn_stage)-1 and j == self.num_dilation+3:
                        x_scaled = x/temp
                    x = self.tcn_stage[i][j](x, training=training)
            outputs.append(x)
        outputs.append(x_scaled)
        return outputs

    def call_logit_only(self, x, training=False, temp=1): # implemented for random logit interpolation in adamatch

        outputs = []
        for i in range(len(self.tcn_stage)):  # loop for
            for j in range(self.num_dilation + 4):
                if 0 < j and j < self.num_dilation + 1:
                    input = x
                    for k in range(5):
                        x = self.tcn_stage[i][j][k](x, training=training)
                    x = input + x
                else:
                    if j == self.num_dilation+3:
                        x_scaled = x/temp
                        outputs.append(x_scaled)
                    x = self.tcn_stage[i][j](x, training=training)
        return outputs

    def predict_classifier(self, X_long, file_boundaries=None):
        if file_boundaries is None:
            file_boundaries = []
        file_boundary_ind = np.where(file_boundaries==1)[0].tolist()
        start = 0  # test_data_start_ind
        if len(file_boundary_ind) > 0:
            if not len(file_boundaries)-1 in file_boundary_ind:
                file_boundary_ind.append(len(file_boundaries)-1)
            for i in file_boundary_ind:
                output_final_file = self.call(X_long[np.newaxis, start:i+1]).numpy()[0]
                if start == 0:
                    output_final = output_final_file
                else:
                    output_final = np.concatenate([output_final, output_final_file], axis=0)
                start = i+1
        else:
            output_final = self.call(X_long[np.newaxis, :, :]).numpy()
        output_final = output_final.reshape((-1, self.num_class))

        return output_final

    def call_emb(self, x, training=True):
        # x = self.input_projection(x)
        for i in range(len(self.tcn_stage)):
            for j in range(self.num_dilation + 4):
                if 0 < j and j < self.num_dilation + 1:
                    input = x
                    for k in range(5):
                        x = self.tcn_stage[i][j][k](x, training=training)
                    x = input + x
                else:
                    if i==len(self.tcn_stage)-1 and j == self.num_dilation+2:
                        break
                    x = self.tcn_stage[i][j](x, training=training)
        for layer in self.proj_layers:
            x = layer(x, training=training)
        return x


    def predict_penultimate(self, X_long, file_boundaries=[]):
        file_boundary_ind = np.where(file_boundaries == 1)[0].tolist()
        if len(file_boundary_ind)==0:
            output_final = self.call_encoder(X_long[np.newaxis, :]).numpy()[0]
        else:
            if not (len(X_long) in file_boundary_ind):
                file_boundary_ind.append(len(X_long))
            # print(file_boundary_ind)
            start = 0
            for i in file_boundary_ind:
                output_final_file = self.call_encoder(X_long[np.newaxis, start:i]).numpy()[0]
                if start == 0:
                    output_final = output_final_file
                else:
                    output_final = np.concatenate([output_final, output_final_file], axis=0)
                start = i
        return output_final


    def predict_logit(self, X_long, file_boundaries=[]):
        file_boundary_ind = np.where(file_boundaries == 1)[0].tolist()
        if len(file_boundary_ind)==0:
            output_final = self.call_logit(X_long[np.newaxis, :]).numpy()[0]
        else:
            if not (len(X_long) in file_boundary_ind):
                file_boundary_ind.append(len(X_long))
            # print(file_boundary_ind)
            start = 0
            for i in file_boundary_ind:
                output_final_file = self.call_logit(X_long[np.newaxis, start:i]).numpy()[0]
                if start == 0:
                    output_final = output_final_file
                else:
                    output_final = np.concatenate([output_final, output_final_file], axis=0)
                start = i
        return output_final

    def call_gradient(self, x, y, training=False):
        '''

        :param x: array-like instance with shape (batch, timestamp, dim)
        :param y: sparse label vector (batch, timestamp, dim=1)
        :return: gradient vector of each timestamp (timestamp, penultimate_dim * num_class)
        '''
        for i in range(len(self.tcn_stage)): # loop for
            for j in range(self.num_dilation + 4):
                if 0 < j and j < self.num_dilation + 1:
                    input = x
                    for k in range(5):
                        x = self.tcn_stage[i][j][k](x, training=training)
                    x = input + x
                else:
                    if i==len(self.tcn_stage)-1 and j == self.num_dilation+1:
                        penultimate_output = x
                    x = self.tcn_stage[i][j](x, training=training)

        cout = tf.reshape(tf.squeeze(x),[x.shape[1],self.num_class,1]) # remove batch axis -> timestamp, num_class
        out = tf.reshape(tf.squeeze(penultimate_output), [x.shape[1],1,penultimate_output.shape[2]]) # timestamp, 1, num_channel
        y = tf.reshape(tf.squeeze(tf.one_hot(y, depth=self.num_class)),[x.shape[1],self.num_class,1])# timestamp, num_class, 1
        dy_dz = cout - y

        cout_np = tf.clip_by_value(cout, clip_value_min=1e-8, clip_value_max=1).numpy().reshape(x.shape[1],self.num_class)
        delta_raw = (tf.math.log(cout_np[1:,:])-tf.math.log(cout_np[:-1,:])).numpy() # timestamp-1, num_class
        delta = tf.math.abs(delta_raw).numpy()
        delta_tilda = tf.clip_by_value(delta, clip_value_min=0, clip_value_max=4).numpy()

        d_delta_dw = 1/cout_np[1:,:]
        d_delta_dw[delta_raw<0] = d_delta_dw[delta_raw<0]*-1
        d_delta_dw[delta>4] = 0
        d_delta_dw = 2 * d_delta_dw * delta_tilda / ((x.shape[1]-1)*self.num_class)
        d_delta_dw_complete = np.zeros((x.shape[1],self.num_class))
        d_delta_dw_complete[1:,:] = d_delta_dw

        d_delta_dw_complete = tf.cast(tf.reshape(tf.constant(d_delta_dw_complete), [x.shape[1],self.num_class,1]),tf.dtypes.float32)
        gradient = tf.reshape(tf.matmul(dy_dz,out) + 0.15*tf.matmul(d_delta_dw_complete,out), [x.shape[1],-1])
        return gradient

    def get_gradient(self, X_long, y_long, file_boundaries):
        file_boundary_ind = np.where(file_boundaries == 1)[0].tolist()
        # file_boundary_ind = file_boundary_ind[file_boundary_ind<=len(X_long)//2].tolist()
        if len(file_boundary_ind)==0:
            output_final = self.call_gradient(X_long[np.newaxis, :], y_long[np.newaxis, :]).numpy()
        else:
            if not (len(X_long) in file_boundary_ind):
                file_boundary_ind.append(len(X_long))
            start = 0
            for i in file_boundary_ind:
                output_final_file = self.call_gradient(X_long[np.newaxis, start:i], y_long[np.newaxis, start:i]).numpy()
                if start == 0:
                    output_final = output_final_file
                else:
                    output_final = np.concatenate([output_final, output_final_file], axis=0)
                start = i
        return output_final


if __name__=="__main__":
    input_length = 256
    num_batch = 32
    dim = 20
    num_class = 10
    total_timestamp = 200000
    file_boundaries = np.zeros(total_timestamp)
    file_boundaries[50000]=1
    file_boundaries[70000]=1
    file_boundaries[120000]=1
    tcn = MSTCN(num_class,dim,is_LLAL=True) # num_class, dim
    print("call", tcn(np.random.rand(num_batch,input_length,dim)).shape)  # batch, timestamp, dim
    # need to call the model at least one time to initialize parameters of the model
    print("call_penul", tcn.call_encoder(np.random.rand(num_batch, input_length, dim)).shape)  # batch, timestamp, dim
    # print("call_training", tcn.call_training(np.random.rand(num_batch,input_length,dim)))  # batch, timestamp, dim

    print("pred_penul", tcn.predict_penultimate(np.zeros((total_timestamp,dim)),file_boundaries).shape)
    for i in range(1):
        print(tcn.train_step(np.random.rand(num_batch,input_length,dim), np.zeros((num_batch,input_length)), np.ones((num_batch,input_length)),curr_epoch=40,file_boundaries=file_boundaries)) # if current epoch is over 80% of total epochs
        print("call_after_training_one_step", np.sum(tf.math.is_nan(tcn(np.random.rand(num_batch,input_length,dim).astype(np.float32))).numpy()))

    g=tcn.call_gradient(np.random.rand(1,total_timestamp,dim), np.random.randint(num_class,size=total_timestamp)[np.newaxis,:])
    G = tcn.get_gradient(np.random.rand(total_timestamp,dim), np.random.randint(num_class,size=total_timestamp), file_boundaries)
    print(g.shape)
    print(G.shape)
    print(tcn.predict_loss(np.zeros((total_timestamp,dim)),file_boundaries).shape)
    print(tcn.get_target_loss(np.random.rand(total_timestamp,dim), np.random.randint(num_class,size=total_timestamp), file_boundaries).shape)
    # tcn.get_LLAL(np.random.rand(1,total_timestamp,dim), np.random.randint(num_class,size=total_timestamp), file_boundaries)
    # print(init_centers(G,40))

    # input_LLAL = tcn.get_input_LLAL(np.random.rand(1,total_timestamp,dim), file_boundaries)
    # output_LLAL = tcn.call_LLAL(input_LLAL)

    tcn.summary()
    # tf.keras.utils.plot_model(tcn, show_shapes=True, to_file="./figures/model_figure.png")
    # rnn = RNN(10,10)
    # print(rnn(np.zeros((num_batch, 5, 10))))
    # rnn.summary()