import os
import sys
import numpy as np
sys.path.append("/home/zebo/git/myRep/Kaggle/Kaggle-DataScience-Bowl/pkugoodspeed/models")
sys.path.append("/home/zebo/git/myRep/Kaggle/Kaggle-DataScience-Bowl/utils")

from process import ImagePrec
from resnet import ResNet
from unet import UNet
from uresnet import UResNet
from opts_parser import getopts

TRAIN_PATH = "/home/zebo/git/myRep/Kaggle/Kaggle-DataScience-Bowl/data/train"
TEST_PATH = "/home/zebo/git/myRep/Kaggle/Kaggle-DataScience-Bowl/data/test"

if __name__ == '__main__':
    C = getopts()
    ip = ImagePrec(path=TRAIN_PATH, size=C['proc']['size'], channel=3, normalize=C['proc']['normalize'])
    n_img = ip.get_num()
    train_x, train_y = ip.get_batch_resized(train_idx=[i for i in range(n_img)])
    if C['augment']:
        train_x, train_y = ip.augment(train_x, train_y)
    # resn = ResNet(input_shape=(C['proc']['size'], C['proc']['size'], 3))
    # resn = UNet(input_shape=(C['proc']['size'], C['proc']['size'], 3))
    resn = UResNet(input_shape=(C['proc']['size'], C['proc']['size'], 3))
    resn.build_model(**C['model_kargs'])
    resn.fit(x=train_x, y=train_y, **C['fit_kargs'])
    model = resn.get_model()
    
    ip.get_test_set(path=TEST_PATH, normalize=C['proc']['normalize'])
    ip.predict_resized(model)
    if not os.path.exists(C['output_dir']):
        os.makedirs(C['output_dir'])
    sub = ip.encoding(threshold=0.5, dilation=True)
    filename = C['output_dir'] + '/subm_dil.csv'
    sub.to_csv(filename, index=False)
    sub = ip.encoding(threshold=0.5, dilation=False)
    filename = C['output_dir'] + '/subm_nodil.csv'
    sub.to_csv(filename, index=False)
    