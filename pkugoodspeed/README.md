### Usage:

- In folder src:

```
$ python singlemodel.py --p params.cfg --m model.cfg
```

The `params.cfg` contains basic information:

- `train_path`: training data path
- `test_path`: testing data path
- `img_size`: the size of cropped images
- `channel`: should always be 3
- `normalize`: whether do normalization or not
- `expand`: how many random crops for each original image

###### `params.cfg` example:
```
{
    "train_path": "/home/zebo/git/myRep/Kaggle/Kaggle-DataScience-Bowl/data/train",
    "test_path": "/home/zebo/git/myRep/Kaggle/Kaggle-DataScience-Bowl/data/test",
    "img_size": 160,
    "channel": 3,
    "normalize": true,
    "expand": 4
}
```

The `model.cfg` contains the model information:

- `model_name`: name of the model
- `learning_rate`: initial learning rate
- `decay_rate`: learning rate decaying factor
- `iterations`: number of iterations
- `epochs`: number of epochs in each iteration
- The layer lists, for `Unet`:
  - `conv_list`: the arguments for convolutional layers
  - `revt_list`: the arguments for up/reverse convolutional layers

###### `model.cfg` example:
```
{
    "model_name": "unet-test",
    "learning_rate": 0.05,
    "decay_rate": 0.9,
    "iterations": 1,
    "epochs": 2,
    "conv_list": [
        {
            "filters": 32,
            "kernel_size": 3,
            "dropout": 0.12
        },
        {
            "filters": 64,
            "kernel_size": 3,
            "dropout": 0.25
        },
        {
            "filters": 128,
            "kernel_size": 3,
            "dropout": 0.5
        },
        {
            "filters": 256,
            "kernel_size": 3,
            "dropout": 0.5
        },
        {
            "filters": 512,
            "kernel_size": 3,
            "dropout": 0.5
        }
    ],
    "revt_list": [
        {
            "filters": 256,
            "kernel_size": 3,
            "dropout": 0.5,
            "cfilters": 256,
            "ckernel_size": 3
        },
        {
            "filters": 128,
            "kernel_size": 3,
            "dropout": 0.5,
            "cfilters": 128,
            "ckernel_size": 3
        },
        {
            "filters": 64,
            "kernel_size": 3,
            "dropout": 0.25,
            "cfilters": 64,
            "ckernel_size": 3
        },
        {
            "filters": 32,
            "kernel_size": 3,
            "dropout": 0.12,
            "cfilters": 32,
            "ckernel_size": 3
        }
    ]
}
```

##### Modules:

- preprocessing code and postprocessing code are in [`utils/process.py`](https://github.com/PKUGoodSpeed/Kaggle-DataScience-Bowl/blob/master/utils/process.py)
- model modules are in [`pkugoodspeed/models/`](https://github.com/PKUGoodSpeed/Kaggle-DataScience-Bowl/tree/master/pkugoodspeed/models)
- main functions' are in [`pkugoodspeed/src/`](https://github.com/PKUGoodSpeed/Kaggle-DataScience-Bowl/tree/master/pkugoodspeed/src)
