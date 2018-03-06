import os
import sys
import numpy as np
sys.path.append("/home/zebo/git/myRep/Kaggle/Kaggle-DataScience-Bowl/pkugoodspeed/models")
sys.path.append("/home/zebo/git/myRep/Kaggle/Kaggle-DataScience-Bowl/utils")

from process import ImagePrec
from unet import UNet
from opts_parser import getopts

if __name__ == '__main__':
    params, model_params = getopts()
    ip = ImagePrec(path=params["train_path"], size=params["img_size"], channel=params["channel"], 
    normalize=params["normalize"], augment=True)
    
    un = UNet(input_shape=(params["img_size"], params["img_size"], params["channel"]))
    un.build_model(conv_list=model_params["conv_list"], revt_list=model_params["revt_list"])
    
    print("\n===================== Start Training ... =====================\n\n")
    lr = model_params["learning_rate"]
    dr = model_params["decay_rate"]
    N_iter = model_params["iterations"]
    model_name = model_params["model_name"]
    for _ in range(N_iter):
        # batch = ip.get_batch_data(expand=params["expand"], seed=None)
        batch = ip.get_batch_resized()
        print batch["x"].shape
        print batch["y"].shape
        un.fit(batch["x"], batch["y"], learning_rate=lr, epochs=model_params['epochs'], check_file=model_name+"_check.h5")
        lr *= dr
    print("\n\n===================== Training finished. =====================\n")
    model_file_path = "./checkpoints/" + model_name + "_check.h5"
    un.load_model(model_file_path)
    
    print("\n===================== Making predictions ... =====================\n\n")
    ip.get_test_set(path=params["test_path"], normalize=params["normalize"])
    model = un.get_model()
    ip.predict(model, stride=params["stride"])
    for thrd in np.arange(0.375, 0.626, 0.025):
        ip.check_results(path="{0}_thrd{1}_output".format(model_name, str(thrd)), threshold=thrd)
        subm = ip.encoding(threshold=thrd, dilation=False)
        subm.to_csv("{0}_thrd{1}_output/nodilu.csv".format(model_name, str(thrd)))
        subm = ip.encoding(threshold=thrd, dilation=True)
        subm.to_csv("{0}_thrd{1}_output/dilute.csv".format(model_name, str(thrd)))