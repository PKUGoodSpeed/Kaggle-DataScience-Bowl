# Basic
import os
import sys
import random
import time
import numpy as np
import pandas as pd

# Image reader
from skimage.io import imread, imshow, imread_collection, concatenate_images, imsave
from skimage.morphology import label
from skimage.transform import resize
from skimage import morphology

# Using multiprocessing
from multiprocessing import Pool

# Visualization for checking results
import matplotlib.pyplot as plt
plt.switch_backend('agg')

class TestModel:
    def predict(self, A):
        return A[:, :, :, 0]

def _run_length_encoding(x, threshold):
    dots = np.where(x.T.flatten() >= threshold)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return ' '.join([str(y) for y in run_lengths])

def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

def rle_decode(mask_rle, shape):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    if type(mask_rle) == str:
        s = mask_rle.split()
    else:
        s = mask_rle
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T

def prob_to_rles(x, cutoff=0.5, dilation=False):
    lab_img = morphology.label(x > cutoff) # split of components goes here
    if dilation:
        for i in range(1, lab_img.max() + 1):    
            lab_img = np.maximum(lab_img, ndimage.morphology.binary_dilation(lab_img==i)*i)
    for i in range(1, lab_img.max() + 1):
        img = lab_img == i
        yield rle_encoding(img)

class ImagePrec:
    _size = None
    _channel = None
    ## Training set info
    _imgs = None
    _ids = None
    _masks = None
    ## Testing set info
    _test_imgs = None
    _test_ids = None
    _test_masks = None

    def __init__(self, path='../data/train', size=128, channel=3, normalize=0):
        """
        It looks that the minimum size of images is (256, 256)
        The channel of the input images are all 4, but only the first 3 matters.
        For the 4th channel, all entries equal 255, which dose not make any differences.
        """
        self._size = size
        self._channel = channel
        self._imgs = []
        self._ids = []
        self._masks = []
        self._n = 0
        
        print("Extracting training image info ...")
        start_time = time.time()
        for img_id in os.listdir(path):
            self._ids.append(img_id)
            img_file_list = os.listdir('{0}/{1}/images'.format(path, img_id))
            assert len(img_file_list) == 1, "Multiple images found in one images id folder."
            assert img_file_list[0] == img_id + '.png', "Image id and image name do not match."
            img = imread('{0}/{1}/images/{1}.png'.format(path, img_id, img_id))[:, :, :channel]
            ## Do normalization
            if normalize==1:
                img = (img.astype(np.float32) - img.mean())/max(1., img.std())
            elif normalize > 1:
                img = img.astype(np.float32)/normalize
                
            mask = np.zeros(img.shape[:2])
            for m in os.listdir('{0}/{1}/masks'.format(path, img_id)):
                mask_ = imread('{0}/{1}/masks/{2}'.format(path, img_id, m))
                assert mask_.shape == img.shape[:2], "Image shape and mask shape do not match."
                mask = np.maximum(mask, mask_)
            mask = mask.astype(np.bool)
            self._imgs.append(img)
            self._masks.append(mask.astype(np.float32))
        self._n = len(self._imgs)
            
        print("Time Usage: {0} sec".format(str(time.time() - start_time)))
        print len(self._ids), len(self._imgs), len(self._masks)
    
    def get_num(self):
        return self._n
    
    def get_ids(self):
        return self._ids

    def get_batch_cropped(self, train_idx, expand=1):
        print("Getting cropped images ...")
        x = []
        y = []
        img_set = [self._imgs[i] for i in train_idx]
        mask_set = [self._masks[i] for i in train_idx]
        for img, mask in zip(img_set, mask_set):
            assert img.shape[0]>=self._size and img.shape[1]>=self._size, "There exist images whose size is smaller than cropped size"
            for k in range(expand):
                i = np.random.randint(img.shape[0] - self._size + 1)
                j = np.random.randint(img.shape[1] - self._size + 1)
                x.append(img[i: i+self._size, j: j+self._size])
                y.append(mask[i: i+self._size, j: j+self._size])
        return np.array(x), np.array(y)
    
    def get_batch_resized(self, train_idx):
        print("Getting resized images ...")
        train_x = np.array([resize(self._imgs[i], (self._size, self._size), mode='constant', preserve_range=True) for i in train_idx])
        tmp_y = np.array([resize(self._masks[i], (self._size, self._size), mode='constant', preserve_range=True) for i in train_idx])
        train_y = np.array([(mask_ >= 0.5).astype(np.float32) for mask_ in tmp_y])
        return train_x, train_y
    
    def augment(self, tr_x, tr_y):
        x_list = [tr_x]
        y_list = [tr_y]
        for _ in range(3):
            x_list.append(np.array([np.rot90(img_) for img_ in x_list[-1]]))
            y_list.append(np.array([np.rot90(mask_) for mask_ in y_list[-1]]))
        for i in range(4):
            x_list.append(np.array([np.flipud(img_) for img_ in x_list[i]]))
            y_list.append(np.array([np.flipud(mask_) for mask_ in y_list[i]]))
        x = np.concatenate(x_list)
        y = np.concatenate(y_list)
        return x, y
    
    def get_valid_set(self, valid_idx):
        self._test_ids = []
        self._test_imgs = []
        
        print("Extracting validation image info ...")
        start_time = time.time()
        self._test_ids = [self._ids[i] for i in valid_idx]
        self._test_imgs = [self._imgs[i] for i in valid_idx]
        print("Time Usage: {0} sec".format(str(time.time() - start_time)))
        print len(self._test_ids), len(self._test_imgs)

    def get_test_set(self, path='../data/test', normalize=0):
        self._test_ids = []
        self._test_imgs = []
        
        print("Extracting testing image info ...")
        start_time = time.time()
        for img_id in os.listdir(path):
            self._test_ids.append(img_id)
            img_file_list = os.listdir('{0}/{1}/images'.format(path, img_id))
            assert len(img_file_list) == 1, "Multiple images found in one images id folder."
            assert img_file_list[0] == img_id + '.png', "Image id and image name do not match."
            img = imread('{0}/{1}/images/{1}.png'.format(path, img_id, img_id))[:, :, :self._channel]
            ## Do normalization
            if normalize == 1:
                img = (img.astype(np.float32) - img.mean())/max(1., img.std())
            elif normalize > 1:
                img = img.astype(np.float32)/normalize
            self._test_imgs.append(img)
        print("Time Usage: {0} sec".format(str(time.time() - start_time)))
        print len(self._test_ids), len(self._test_imgs)

    def predict(self, model, stride=16):
        """
        Generate the probability map
        """
        print "Getting predictions ..."
        start_time = time.time()
        self._test_masks = []
        for img in self._test_imgs:
            coordinates = [
                (min(i, img.shape[0]-self._size), min(j, img.shape[1]-self._size))
                for i in range(0, img.shape[0], stride)
                for j in range(0, img.shape[1], stride)
            ]
            test_batch = model.predict(np.array([img[i: i+self._size, j: j+self._size] for i, j in coordinates]))
            mask = np.zeros(img.shape[:2])
            ratio = np.zeros(img.shape[:2])
            for (i, j), _mask in zip(coordinates, test_batch):
                mask[i: i+self._size, j: j+self._size] += _mask
                ratio[i: i+self._size, j: j+self._size] += np.ones((self._size, self._size))
            self._test_masks.append(mask.astype(np.float32)/ratio.astype(np.float32))
        print("Time Usage: {0} sec".format(str(time.time() - start_time)))
        return self._test_masks
    
    def predict_resized(self, model):
        """
        Generate the probability map
        """
        print "Getting predictions for resized input ..."
        start_time = time.time()
        self._test_masks = []
        test_X = np.array([resize(img, (self._size, self._size), mode='constant', preserve_range=True) for img in self._test_imgs])
        test_Y = model.predict(test_X)
        for pred, img in zip(test_Y, self._test_imgs):
            self._test_masks.append(resize(pred, (img.shape[0], img.shape[1]), mode='constant', preserve_range=True))
        print("Time Usage: {0} sec".format(str(time.time() - start_time)))
        return self._test_masks

    def save_predictions(self, path='data/predictions'):
        """
        Saving OOF features and test predictions
        """
        print("Saving predictions ...")
        if not os.path.exists(path):
            os.makedirs(path)
        for idx, pred in zip(self._test_ids, self._test_masks):
            filename = path + '/' + idx + ".png"
            imsave(filename, pred)
        
    def check_results(self, path="./output", threshold=0.5):
        """ Check how good the predictions are """
        idx = np.array([np.random.randint(len(self._test_imgs)) for i in range(9)])
        imgs = [self._test_imgs[i] for i in idx]
        masks = [(self._test_masks[i]>threshold) for i in idx]
        
        if not os.path.exists(path):
            os.system("mkdir {0}".format(path))
        
        fig, axes = plt.subplots(3, 3, figsize = (12, 12))
        fig.subplots_adjust(hspace = 0.3, wspace = 0.3)
        for i, ax in enumerate(axes.flat):
            ax.imshow(imgs[i])
            ax.set_xticks([])
            ax.set_yticks([])
        plt.savefig(path+"/imgs.png")
        print("Images are show in {0}/imgs.png".format(path))
        
        fig, axes = plt.subplots(3, 3, figsize = (12, 12))
        fig.subplots_adjust(hspace = 0.3, wspace = 0.3)
        for i, ax in enumerate(axes.flat):
            ax.imshow(masks[i])
            ax.set_xticks([])
            ax.set_yticks([])
        plt.savefig(path+"/masks.png")
        print("Masks are show in {0}/masks.png".format(path))

    def encoding(self, threshold=0.5, dilation=False):
        new_test_ids = []
        rles = []
        for id_, pred_mask in zip(self._test_ids, self._test_masks):
            rle = list(prob_to_rles(pred_mask, cutoff=threshold, dilation=dilation))
            rles.extend(rle)
            new_test_ids.extend([id_] * len(rle))
        sub = pd.DataFrame()
        sub['ImageId'] = new_test_ids
        sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
        return sub