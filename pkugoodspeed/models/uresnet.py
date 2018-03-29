import os
import numpy as np
from keras.models import Model
from keras.optimizers import SGD, Adam
from keras.callbacks import LearningRateScheduler, Callback, EarlyStopping, ModelCheckpoint
from keras.layers import Input, Conv2D, Conv2DTranspose, Activation, Dropout, Reshape
from keras.layers import MaxPooling2D, AveragePooling2D, concatenate
from keras.layers.core import Lambda
from model_utils import *


global_learning_rate = 0.01
global_decaying_rate = 0.92

class UResNet:
    _input_shape = (256, 256, 3)
    _output_shape = (256, 256)
    _model = None
    
    def __init__(self, input_shape):
        self._input_shape = input_shape
        self._output_shape = input_shape[:2]
    
    def build_model(self, init_channel=16, depth=4, 
    kernel_size=3, initial_dropout=0.1, activation='relu', smooth=3):
        in_layer = Input(self._input_shape)
        k_size = kernel_size
        n_channel = init_channel
        dropout_ratio = initial_dropout
        tmp = in_layer
        conv = []
        drops = []
        for i in range(depth):
            tmp1 = Conv2D(filters=n_channel, kernel_size=k_size, activation=activation,
            kernel_initializer='he_normal', strides=1, padding="same") (tmp)
            tmp1 = Dropout(dropout_ratio) (tmp1)
            tmp = concatenate([tmp, tmp1])
            tmp = Conv2D(filters=n_channel, kernel_size=k_size, activation=activation,
            kernel_initializer='he_normal', strides=1, padding="same") (tmp)
            tmp = Dropout(dropout_ratio) (tmp)
            if i < depth-1:
                drops.append(dropout_ratio)
                dropout_ratio = min(dropout_ratio*1.7, 0.4)
                conv.append(tmp)
                tmp = MaxPooling2D((2, 2)) (tmp)
                n_channel *= 2
        
        for i in range(1, depth):
            n_channel /= 2
            tmp = Conv2DTranspose(filters=n_channel, kernel_size=(2, 2), strides=(2, 2),
            padding="same") (tmp)
            tmp = concatenate([tmp, conv[-i]])
            tmp1 = Conv2D(filters=n_channel, kernel_size=k_size, activation=activation,
            kernel_initializer='he_normal', strides=1, padding="same") (tmp)
            tmp = Dropout(drops[-i]) (tmp1)
            tmp = concatenate([tmp, tmp1])
            tmp = Conv2D(filters=n_channel, kernel_size=k_size, activation=activation,
            kernel_initializer='he_normal', strides=1, padding="same") (tmp)
            tmp = Dropout(drops[-i]) (tmp)
        
        out_layer = Conv2D(filters=1, kernel_size=smooth, activation='sigmoid', 
        strides=1, padding="same") (tmp)
        out_layer = Reshape(self._output_shape) (out_layer)

        # construct model
        self._model = Model(inputs=[in_layer], outputs=[out_layer])
        self._model.summary()
    
    def get_model(self):
        return self._model

    def fit(self, x, y, learning_rate=0.02, decaying_rate=0.9, epochs=2, loss='bin_cross', check_file='weights.h5'):
        self._model.compile(optimizer='sgd', loss=loss_map[loss], metrics=[mean_iou, 'accuracy'])
        global global_learning_rate
        global global_decaying_rate
        ## Setting learning rate explicitly
        global_learning_rate = learning_rate
        global_decaying_rate = decaying_rate
        def scheduler(epoch):
            global global_learning_rate
            global global_decaying_rate
            global_learning_rate *= global_decaying_rate
            print("CURRENT LEARNING RATE = " + str(global_learning_rate))
            return global_learning_rate
        change_lr = LearningRateScheduler(scheduler)
        
        ## Set early stopper:
        earlystopper = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')
        
        ## Set Check point
        if not os.path.exists('./checkpoints'):
            os.system('mkdir checkpoints')
        checkpointer = ModelCheckpoint(filepath='./checkpoints/'+check_file, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
        
        history = self._model.fit(x, y, batch_size=32, epochs=epochs, verbose=1,
        validation_split=0.1, callbacks=[checkpointer, change_lr])
        ## self._model.load_weights("./checkpointer/" + check_file)
        return history

    def load_model(self, filepath):
        self._model.load_weights(filepath)