import os
import sys
import numpy as np
from sklearn.cross_validation import KFold
sys.path.append("/home/zebo/git/myRep/Kaggle/Kaggle-DataScience-Bowl/pkugoodspeed/models")
sys.path.append("/home/zebo/git/myRep/Kaggle/Kaggle-DataScience-Bowl/utils")

from process import ImagePrec
from resnet import ResNet
from unet import UNet
from uresnet import UResNet
from opts_parser import getopts

TRAIN_PATH = "/home/zebo/git/myRep/Kaggle/Kaggle-DataScience-Bowl/data/train"
TEST_PATH = "/home/zebo/git/myRep/Kaggle/Kaggle-DataScience-Bowl/data/test"

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

if __name__ == '__main__':
    C = getopts()
    ip = ImagePrec(path=TRAIN_PATH, size=C['proc']['size'], channel=3, normalize=C['proc']['normalize'])
    n_img = ip.get_num()
    # resn = ResNet(input_shape=(C['proc']['size'], C['proc']['size'], 3))
    # resn = UNet(input_shape=(C['proc']['size'], C['proc']['size'], 3))
    resn = UResNet(input_shape=(C['proc']['size'], C['proc']['size'], 3))
    kf = KFold(n_img, n_folds=5, random_state=414)
    for i, (train_index, valid_index) in enumerate(kf):
        resn.build_model(**C['model_kargs'])
        train_x, train_y = ip.get_batch_resized(train_idx=train_index)
        valid_x, valid_y = ip.get_batch_resized(train_idx=valid_index)
        if C['augment']:
            train_x, train_y = ip.augment(train_x, train_y)
        resn.fit(x=train_x, y=train_y, valid_set=(valid_x, valid_y),
        check_file='resz_weights_fd{0}.h5'.format(str(i)), **C['fit_kargs'])
        model = resn.get_model()
        ip.get_valid_set(valid_idx=valid_index)
        ip.predict_resized(model)
        ip.save_predictions(path='../output/prob_map/uresnet_resz/oof')
    
    resn.build_model(**C['model_kargs'])
    train_x, train_y = ip.get_batch_resized(train_idx=[i for i in range(n_img)])
    resn.fit(x=train_x, y=train_y, valid_set=None, check_file='resz_weights_test.h5', **C['fit_kargs'])
    model = resn.get_model()
    
    ip.get_test_set(path=TEST_PATH, normalize=C['proc']['normalize'])
    ip.predict_resized(model)
    ip.save_predictions(path='../output/prob_map/uresnet_resz/test_pred')
    if not os.path.exists(C['output_dir']):
        os.makedirs(C['output_dir'])
    sub = ip.encoding(threshold=0.5, dilation=False)
    filename = C['output_dir'] + '/subm_nodil.csv'
    sub.to_csv(filename, index=False)
    sub = ip.encoding(threshold=0.5, dilation=True)
    filename = C['output_dir'] + '/subm_dil.csv'
    sub.to_csv(filename, index=False)
    