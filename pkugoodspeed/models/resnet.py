import os
import numpy as np
from keras.models import Model
from keras.optimizers import SGD, Adam
from keras.callbacks import LearningRateScheduler, Callback, EarlyStopping, ModelCheckpoint
from keras.layers import Input, Conv2D, Activation, Dropout, Lambda, Dense, BatchNormalization
from keras.layers import MaxPooling2D, AveragePooling2D, concatenate, Add, Reshape
from keras.layers import GlobalMaxPooling2D, GlobalAveragePooling2D, merge, Flatten
from model_utils import *


global_learning_rate = 0.01
global_decaying_rate = 0.92

class ResNet:
    _input_shape = (256, 256, 3)
    _output_shape = (256, 256)
    _model = None
    
    def __init__(self, input_shape):
        self._input_shape = input_shape
        self._output_shape = input_shape[:2]
    
    def build_model(self, n_filters=16, depth=4, res_num=3):
        in_layer = Input(self._input_shape)
        
        kernel = BatchNormalization(axis=-1) (Conv2D(filters=n_filters, kernel_size=3, padding="same") (in_layer))
        kernel = Activation('elu') (kernel)
        
        for _ in range(depth):
            layerA = Conv2D(filters=2*n_filters, kernel_size=1, padding="same") (kernel)
            layerB = Conv2D(filters=int(n_filters/2), kernel_size=1, padding="same") (kernel)
            layerB = Activation('elu') (BatchNormalization(axis=-1) (layerB))
            layerB = Conv2D(filters=int(n_filters/2), kernel_size=3, padding="same") (layerB)
            layerB = Activation('elu') (BatchNormalization(axis=-1) (layerB))
            layerB = Conv2D(filters=2*n_filters, kernel_size=1, padding="same") (layerB)
            kernel = Add() ([layerA, layerB])
            kernel = Dropout(0.15) (kernel)
            
            for _ in range(res_num):
                layerB = Activation('elu') (BatchNormalization(axis=-1) (kernel))
                layerB = Conv2D(filters=int(n_filters/2), kernel_size=1, padding="same") (layerB)
                layerB = Activation('elu') (BatchNormalization(axis=-1) (layerB))
                layerB = Conv2D(filters=int(n_filters/2), kernel_size=3, padding="same") (layerB)
                layerB = Activation('elu') (BatchNormalization(axis=-1) (layerB))
                layerB = Conv2D(filters=2*n_filters, kernel_size=1, padding="same") (layerB)
                kernel = Add() ([kernel, layerB])
                kernel = Dropout(0.15) (kernel)
            
            kernel = Activation('elu') (BatchNormalization(axis=-1) (kernel))
            kernel = Dropout(0.4) (kernel)

            out_layer = Conv2D(filters=1, kernel_size=1, activation='sigmoid', 
            strides=1, padding="same") (kernel)

            out_layer = Reshape(self._output_shape) (out_layer)

        # construct model
        self._model = Model(inputs=[in_layer], outputs=[out_layer])
        self._model.summary()
    
    def get_model(self):
        return self._model

    def fit(self, x, y, valid_set=None, learning_rate=0.02, decaying_rate=0.9, epochs=2, loss='bin_cross', check_file='weights.h5'):
        self._model.compile(optimizer=Adam(0.001), loss=loss_map[loss], metrics=[mean_iou, 'accuracy'])
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
        earlystopper = EarlyStopping(monitor='val_acc', patience=5, verbose=1, mode='auto')
        
        ## Set Check point
        if not os.path.exists('./checkpoints'):
            os.system('mkdir checkpoints')
        checkpointer = ModelCheckpoint(filepath='./checkpoints/'+check_file, monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
        
        history = self._model.fit(x, y, batch_size=8, epochs=epochs, verbose=1,
        validation_data=valid_set, callbacks=[earlystopper, checkpointer, change_lr])
        ## self._model.load_weights("./checkpointer/" + check_file)
        return history

    def load_model(self, filepath):
        self._model.load_weights(filepath)