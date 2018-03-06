import numpy as np
import tensorflow as tf

class TensorflowUnet:
    """ Implement a U-net network, which is slolen from:
    https://www.kaggle.com/raoulma/nuclei-dsb-2018-tensorflow-u-net-score-0-352"""
    
    self._input_shape = (128, 128, 3)
    self._output_shape = (128, 128)
    self._params = {}
    self._learning_rate = 0.01
    self._decay_rate = 0.85
    
    
    def __init__(self, input_shape, output_shape, )
    
    
    def learning_rate_decay(self):
        """ Adaptive Learning Rate """
        self._learning_rate *= self._decay_rate

    def get_weights(self, shape, name=None):
        """ Weights initialization """
        initializer = tf.contrib.layers.xavier_initializer
        #initializer = tf.truncated_normal(shape, stddev=0.1)
        #initializer = tf.contrib.layers.variance_scaling_initializer()
        return tf.get_variable(name, shape=shape, initializer=initializer)

    def get_biases(self, shape, name=None):
        """ Biases initialization """
        initializer = tf.contrib.layers.xavier_initializer
        #initializer = tf.truncated_normal(shape, stddev=0.1)
        #initializer = tf.contrib.layers.variance_scaling_initializer()
        return tf.get_variable(name, shape=shape, initializer=initializer)

    def conv2d(self, x, W, name=None):
        """ Get conv2d layer """
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME", name=name)

    def max_pool_2x2(self, x, name=None):
        """ Max Pooling 2x2 """
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name=name)

    def conv2d_transpose(self, x, filters, name=None):
        """ Transpose 2d Convolution. """
        return tf.layers.conv2d_transpose(x, filters=filters, kernel_size=(2, 2), strides=(2, 2), padding="SAME")
    
    def leaky_relu(self, z, beta=0.01, name=None):
        return tf.maximum(beta*z, z, name=name)
    
    def elu(self, x, name=None):
        return tf.nn.elu(x, name=name)
    
    def loss_tensor(self, y, y_pred, dice_loss=1):
        """ Define loss """
        if dice_loss:
            axis=np.arange(1,len(self.output_shape)+1)
            offset = 1e-6
            # Dice loss based on Jaccard dice score coefficent.
            corr = tf.reduce_sum(y * y_pred, axis=axis)
            l2_pred = tf.reduce_sum(tf.square(y_pred), axis=axis)
            l2_true = tf.reduce_sum(tf.square(y), axis=axis)
            dice_coeff = (2. * corr + offset) / (l2_true + l2_pred + offset)

            if dice_loss == 2:
                corr_inv = tf.reduce_sum((1.-y) * (1.-y_pred), axis=axis)
                l2_pred_inv = tf.reduce_sum(tf.square(1.-y_pred), axis=axis)
                l2_true_inv = tf.reduce_sum(tf.square(1.-y), axis=axis)
                dice_coeff = ((corr + offset) / (l2_true + l2_pred + offset) +
                (corr_inv + offset) / (l2_pred_inv + l2_true_inv + offset))
            
            loss = tf.subtract(1., tf.reduce_mean(dice_coeff))

        else:
            # This loss must be computed before sigmoid activations
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=y, logits=y_pred))

    def optimizer_tensor(self, loss):
        """ Optimizer tensor """
        return tf.train.AdamOptimizer(self._learning_rate).minimize(loss, name='adam')

    def batch_norm(self, x, name=None):
        return tf.layers.batch_normalization(x, momentum=0.9, name=name)
    
    def dropout(self, x, ratio, name=None):
        return tf.layers.dropout(x, ratio, name=name)
    
    def num_of_weights(self, tensors):
        """ Compute the total number of weights """
        sum_= 0
        for i in range(len(tensors)):
            m = 1
            for j in range(len(tensors[i].shape)):
              m *= int(tensors[i].shape[j])
            sum_ += m
        return sum_

    def get_Unet(self, n_depth, filters=[]):
        height = self.input_shape[0]
        width = self.input_shape[1]
        channel = self.input_shape[2]
        for i in range(n_depth):
            with tf.name_scope("unit_{0}".format(str(i))):
                
            
        
        with tf.name_scope('unit1'):
            W1_1 = self.get_weights([3, 3, self.input_shape[2], 16], 'W1_1')