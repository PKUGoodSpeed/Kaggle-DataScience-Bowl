import os
import sys
import numpy as np
sys.path.append("/home/zebo/git/myRep/Kaggle/Kaggle-DataScience-Bowl/pkugoodspeed/models")
sys.path.append("/home/zebo/git/myRep/Kaggle/Kaggle-DataScience-Bowl/utils")

from process import ImagePrec
from resnet import ResNet
from unet import UNet
from opts_parser import getopts

TRAIN_PATH = "/home/zebo/git/myRep/Kaggle/Kaggle-DataScience-Bowl/data/train"
TEST_PATH = "/home/zebo/git/myRep/Kaggle/Kaggle-DataScience-Bowl/data/test"

if __name__ == '__main__':
    C = getopts()
    ip = ImagePrec(path=TRAIN_PATH, size=C['proc']['size'], channel=3, normalize=C['proc']['normalize'])
    n_img = ip.get_num()
    train_x, train_y = ip.get_batch_resized(train_idx=[i for i in range(n_img)])
    train_x, train_y = ip.augment(train_x, train_y)
    # resn = ResNet(input_shape=(C['proc']['size'], C['proc']['size'], 3))
    resn = UNet(input_shape=(C['proc']['size'], C['proc']['size'], 3))
    resn.build_model(**C['model_kargs'])
    resn.fit(x=train_x, y=train_y, **C['fit_kargs'])
    model = resn.get_model()
    
    ip.get_test_set(path=TEST_PATH, normalize=C['proc']['normalize'])
    ip.predict_resized(model)
    if not os.path.exists(C['output_dir']):
        os.makedirs(C['output_dir'])
    for thred in np.arange(0.46, 0.55, 0.02):
        sub = ip.encoding(threshold=thred)
        filename = C['output_dir'] + '/subm_' + str(thred) +'.csv' 
        sub.to_csv(filename, index=False)
    