import os
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.optimizers import SGD, Adam
from keras.callbacks import LearningRateScheduler, Callback, EarlyStopping, ModelCheckpoint
from keras.layers import Input, Conv2D, Conv2DTranspose, Activation, Dropout, Reshape
from keras.layers import MaxPooling2D, AveragePooling2D, concatenate
from keras import backend as K

global_learning_rate = 0.01

def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred>t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2, y_true)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)

class UNet:
    _input_shape = (128, 128, 3)
    _output_shape = None
    _model = None
    
    def __init__(self, input_shape):
        self._input_shape = input_shape
        self._output_shape = input_shape[:2]
    
    def build_model(self, conv_list, revt_list):
        assert len(conv_list) == len(revt_list) + 1
        depth = len(conv_list)
        in_layer = Input(self._input_shape)
        conv = []
        pool = []
        revt = []
        tmp = in_layer
        for i, c in enumerate(conv_list):
            tmp = Conv2D(filters=c["filters"], kernel_size=c["kernel_size"], activation="elu",
            kernel_initializer='he_normal', strides=1, padding="same") (tmp)
            tmp = Dropout(c["dropout"]) (tmp)
            tmp = Conv2D(filters=c["filters"], kernel_size=c["kernel_size"], activation="elu",
            kernel_initializer='he_normal', strides=1, padding="same") (tmp)
            conv.append(tmp)
            if i < depth-1:
                tmp = MaxPooling2D((2, 2)) (tmp)
            ## tmp = AveragePooling2D((2, 2)) (tmp)
        
        for i, r in enumerate(revt_list):
            tmp = Conv2DTranspose(filters=r["filters"], kernel_size=r["kernel_size"], strides=(2, 2),
            padding="same") (tmp)
            tmp = concatenate([tmp, conv[depth-i-2]])
            tmp = Conv2D(filters=r["cfilters"], kernel_size=r["ckernel_size"], activation="elu",
            kernel_initializer='he_normal', strides=1, padding="same") (tmp)
            tmp = Dropout(r["dropout"]) (tmp)
            tmp = Conv2D(filters=r["cfilters"], kernel_size=r["ckernel_size"], activation="elu",
            kernel_initializer='he_normal', strides=1, padding="same") (tmp)
        
        out_layer = Reshape(self._output_shape) (Conv2D(1, (1,1), activation="sigmoid") (tmp))

        # construct model
        self._model = Model(inputs=[in_layer], outputs=[out_layer])
        self._model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[mean_iou, 'accuracy'])
        self._model.summary()
    
    def get_model(self):
        return self._model

    def fit(self, x, y, learning_rate=0.02, epochs=2, check_file='weights.h5'):
        global global_learning_rate
        ## Setting learning rate explicitly
        global_learning_rate = learning_rate
        def scheduler(epoch):
            global global_learning_rate
            if epoch == 0:
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
        validation_split=0.1, callbacks=[earlystopper, checkpointer, change_lr])
        ## self._model.load_weights("./checkpointer/" + check_file)
        return history

    def load_model(self, filepath):
        self._model.load_weights(filepath)