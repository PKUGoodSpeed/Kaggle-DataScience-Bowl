import os
import sys
import numpy as np
from sklearn.cross_validation import KFold
sys.path.append("/home/zebo/git/myRep/Kaggle/Kaggle-DataScience-Bowl/pkugoodspeed/models")
sys.path.append("/home/zebo/git/myRep/Kaggle/Kaggle-DataScience-Bowl/utils")

from process import ImagePrec
from resnet import ResNet
from unet import UNet
# from uresnet import UResNet
from opts_parser import getopts

TRAIN_PATH = "/home/zebo/git/myRep/Kaggle/Kaggle-DataScience-Bowl/data/train"
TEST_PATH = "/home/zebo/git/myRep/Kaggle/Kaggle-DataScience-Bowl/data/test"

if __name__ == '__main__':
    C = getopts()
    ip = ImagePrec(path=TRAIN_PATH, size=C['proc']['size'], channel=3, normalize=C['proc']['normalize'])
    n_img = ip.get_num()
    resn = ResNet(input_shape=(C['proc']['size'], C['proc']['size'], 3))
    # resn = UNet(input_shape=(C['proc']['size'], C['proc']['size'], 3))
    # resn = UResNet(input_shape=(C['proc']['size'], C['proc']['size'], 3))
    
    resn.build_model(**C['model_kargs'])
    lr = C['fit_kargs']['learning_rate']
    dr = C['fit_kargs']['decaying_rate']
    ls = C['fit_kargs']['loss']
    print("Starting Producing test predictions ...")
    all_idx = np.random.permutation([i for i in range(n_img)])
    n_train = int(0.9 * n_img)
    train_idx = all_idx[: n_train]
    valid_idx = all_idx[n_train: ]
    
    for _ in range(C['fit_kargs']['epochs']):
        train_x, train_y = ip.get_batch_cropped(train_idx=train_idx, expand=2)
        valid_x, valid_y = ip.get_batch_cropped(train_idx=valid_idx, expand=1)
        if C['augment']:
            train_x, train_y = ip.augment(train_x, train_y)
        resn.fit(x=train_x, y=train_y, valid_set=(valid_x, valid_y), learning_rate=lr, 
        decaying_rate=1., epochs=1, loss=ls, check_file='weights_test.h5')
        lr *= dr

    resn.load_model('./checkpoints/weights_test.h5')
    model = resn.get_model()
    ip.get_test_set(path=TEST_PATH, normalize=C['proc']['normalize'])
    ip.predict(model, stride=16)
    ip.save_predictions(path='../output/prob_map/resnet_crop/test_pred')
    if not os.path.exists(C['output_dir']):
        os.makedirs(C['output_dir'])
    thres = [0.50, 0.525, 0.55, 0.575, 0.60]
    for th in thres:
        sub = ip.encoding(threshold=th, dilation=False)
        filename = C['output_dir'] + '/subm_{th}.csv'.format(th=str(th))
        sub.to_csv(filename, index=False)