import random

import numpy as np

import skimage
from skimage.exposure import equalize_adapthist as clahe

def hwc_to_chw(img):
    return np.transpose(img, axes=[2, 0, 1])


def resize_and_crop(pilimg, scale=0.5, final_height=None):
    w = pilimg.size[0]
    h = pilimg.size[1]
    newW = int(w * scale)
    newH = int(h * scale)

    if not final_height:
        diff = 0
    else:
        diff = newH - final_height

    img = pilimg.resize((newW, newH))
    img = img.crop((0, diff // 2, newW, newH - diff // 2))
    ar = np.array(img, dtype=np.float32)
    if len(ar.shape) == 2:
        # for greyscale images, add a new axis
        ar = np.expand_dims(ar, axis=2)
    return ar

def batch(iterable, batch_size):
    """Yields lists by batch"""
    b = []
    for i, t in enumerate(iterable):
        b.append(t)
        if (i + 1) % batch_size == 0:
            yield b
            b = []

    if len(b) > 0:
        yield b


def split_train_val(dataset, val_percent=0.05):
    dataset = list(dataset)
    length = len(dataset)
    n = int(length * val_percent)
    random.shuffle(dataset)
    return {'train': dataset[:-n], 'val': dataset[-n:]}


## better normalize
def normalize(x,method='clahe'):
    immin = np.min(x)
    immax = np.max(x)
    if (immax > immin):
        if (method=='clahe'):
            x2 = ((x-immin) / (immax-immin))
            for i in range(x2.shape[0]):
                clares = clahe(x2[i,:,:])
                x2[i,:,:] = clares
            return (x2)
        else:
            return ((x-np.min(x)) / (np.max(x)-np.min(x)))
    else:
        return(x) ## constant 


# credits to https://stackoverflow.com/users/6076729/manuel-lagunas
def rle_encode(mask_image):
    pixels = mask_image.flatten()
    # We avoid issues with '1' at the start or end (at the corners of
    # the original image) by setting those pixels to '0' explicitly.
    # We do not expect these to be non-zero for an accurate mask,
    # so this should not harm the score.
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    return runs
