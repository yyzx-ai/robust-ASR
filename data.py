import tensorflow as tf
import tf2lib as tl
import numpy as np
from python_speech_features import mfcc
import scipy.io.wavfile as wavread
import numpy as np
import os
import imlib as im
import pylib as py
#print(t.shape)
# ==============================================================================
# =                                  datasets                                  =
# ==============================================================================

def make_32x32_dataset(dataset, batch_size, drop_remainder=True, shuffle=True, repeat=1):
    if dataset == 'mnist':
        (train_images, _), (_, _) = tf.keras.datasets.mnist.load_data()
        train_images.shape = train_images.shape + (1,)
    elif dataset == 'fashion_mnist':
        (train_images, _), (_, _) = tf.keras.datasets.fashion_mnist.load_data()
        train_images.shape = train_images.shape + (1,)
    elif dataset == 'cifar10':
        (train_images, _), (_, _) = tf.keras.datasets.cifar10.load_data()
    else:
        raise NotImplementedError
    f2=open('原始样本.txt','w')
    f2.write(str(train_images))
    f2.close()
    @tf.function
    def _map_fn(img):
        img = tf.image.resize(img, [32, 32])
        img = tf.clip_by_value(img, 0, 255)
        img = img / 127.5 - 1
        return img

    dataset = tl.memory_data_batch_dataset(train_images,
                                           batch_size,
                                           drop_remainder=drop_remainder,
                                           map_fn=_map_fn,
                                           shuffle=shuffle,
                                           repeat=repeat)
    img_shape = (32, 32, train_images.shape[-1])
    len_dataset = len(train_images) // batch_size

    return dataset, img_shape, len_dataset
#自制----------------------------------------------------------------------------------------《《《《《
def make_self_dataset( path,batch_size, drop_remainder=True, shuffle=True, repeat=1):
    def mfcc_(filename):
        fs,data=wavread.read(filename)

        feature=mfcc(data, samplerate=fs,numcep=32,nfilt=64,winfunc=np.hamming)

        return feature
    #print(os.getcwd())
    s_t=104#语音长度相关（时域常量）
    wav_data=os.listdir(path)
    t=np.zeros([len(wav_data),s_t,32],dtype=float)
    for i in range(len(wav_data)):
        mf=mfcc_(path+r'/'+wav_data[i])
        t[i][:mf.shape[0]]=mf[:]
    f3=open('原始样本_0.txt','w')
    f3.write(str(t))
    f3.close()
    #(train_images, _), (_, _) = tf.keras.datasets.mnist.load_data()
    train_images=t
    #for i in range(100):
        
        #im.imwrite(train_images[i]/255, py.join('temp/', '%s.jpg'%i))
    #print('原始图片尺寸：{}\n类型：{}'.format(train_images.shape,type(train_images)))
    train_images.shape = train_images.shape + (1,)
    f2=open('原始样本.txt','w')
    f2.write(str(train_images))
    f2.close() 
    #img = im.immerge(train_images, n_rows=10).squeeze()
    #im.imwrite(train_images, py.join('temp/', '2.jpg'))
    #print('原始图片尺寸：{}\n类型：{}'.format(train_images.shape,type(train_images)))
    @tf.function
    def _map_fn(img):
        #img = tf.image.resize(img, [32, 32])
        #img = tf.clip_by_value(img, 0, 255)
        #img = img / 127.5 - 1
        img=img/255
        return img
    #print('修改后图片尺寸：{}\n'.format(_map_fn(img).shape))
    dataset = tl.memory_data_batch_dataset(train_images,
                                           batch_size,
                                           drop_remainder=drop_remainder,
                                           map_fn=_map_fn,
                                           shuffle=shuffle,
                                           repeat=repeat)
    img_shape = (s_t, 32, train_images.shape[-1])
    #print(img_shape)
    len_dataset = len(train_images) // batch_size

    return dataset, img_shape, len_dataset

#--------------------------------------------------------------------------------------------------------------
def make_celeba_dataset(img_paths, batch_size, resize=64, drop_remainder=True, shuffle=True, repeat=1):
    @tf.function
    def _map_fn(img):
        crop_size = 108
        img = tf.image.crop_to_bounding_box(img, (218 - crop_size) // 2, (178 - crop_size) // 2, crop_size, crop_size)
        img = tf.image.resize(img, [resize, resize])
        img = tf.clip_by_value(img, 0, 255)
        img = img / 127.5 - 1
        return img

    dataset = tl.disk_image_batch_dataset(img_paths,
                                          batch_size,
                                          drop_remainder=drop_remainder,
                                          map_fn=_map_fn,
                                          shuffle=shuffle,
                                          repeat=repeat)
    img_shape = (resize, resize, 3)
    len_dataset = len(img_paths) // batch_size

    return dataset, img_shape, len_dataset


def make_anime_dataset(img_paths, batch_size, resize=64, drop_remainder=True, shuffle=True, repeat=1):
    @tf.function
    def _map_fn(img):
        img = tf.image.resize(img, [resize, resize])
        img = tf.clip_by_value(img, 0, 255)
        img = img / 127.5 - 1
        return img

    dataset = tl.disk_image_batch_dataset(img_paths,
                                          batch_size,
                                          drop_remainder=drop_remainder,
                                          map_fn=_map_fn,
                                          shuffle=shuffle,
                                          repeat=repeat)
    img_shape = (resize, resize, 3)
    len_dataset = len(img_paths) // batch_size

    return dataset, img_shape, len_dataset


# ==============================================================================
# =                               custom dataset                               =
# ==============================================================================

def make_custom_datset(img_paths, batch_size, resize=64, drop_remainder=True, shuffle=True, repeat=1):
    @tf.function
    def _map_fn(img):
        # ======================================
        # =               custom               =
        # ======================================
        img = ...  # custom preprocessings, should output img in [0.0, 255.0]
        # ======================================
        # =               custom               =
        # ======================================
        img = tf.image.resize(img, [resize, resize])
        img = tf.clip_by_value(img, 0, 255)
        img = img / 127.5 - 1
        return img

    dataset = tl.disk_image_batch_dataset(img_paths,
                                          batch_size,
                                          drop_remainder=drop_remainder,
                                          map_fn=_map_fn,
                                          shuffle=shuffle,
                                          repeat=repeat)
    img_shape = (resize, resize, 3)
    len_dataset = len(img_paths) // batch_size

    return dataset, img_shape, len_dataset
