# coding = utf-8
# GAN の学習には encoded_noize しか使わなくした版
# E+G (G固定) → D → E+G+D (E, D 固定) で学習する

from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense, Activation, Reshape
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D, Conv2DTranspose
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Flatten, Dropout
from keras.initializers import RandomNormal as rand
from keras.initializers import TruncatedNormal as trunc
import math
import numpy as np
import os
import random
from keras.optimizers import Adam

import models.FriendsLoader as FriendsLoader

import cv2
import dill

from setting import IMG_SIZE
from setting import NOIZE_SIZE
from setting import BATCH_SIZE
from setting import KERNEL_CORE_SIZE
from setting import START_EPOCH
from setting import NUM_EPOCH
from setting import SPAN_CHECKPOINT
from setting import SPAN_UPDATE_NOIZE

from setting import USE_DATA_RATE

from setting import NEXT_PATTERN

from setting import D_LR
from setting import D_BETA
from setting import G_LR
from setting import G_BETA
from setting import E_LR
from setting import E_BETA

from setting import STDDEV

from setting import BN_M
from setting import BN_E

from setting import D_LEARNING_STEP

from models.InputManager import InputManager

SAVE_MODEL_PATH = "tmp/save_models/"
SAVE_NOIZE_PATH = "tmp/save_noizes/"
GENERATED_IMAGE_PATH = "tmp/"

def denseLayer(filters, init="he_normal", input_shape=None):
    if input_shape is not None:
       return Dense(
                filters,                 \
                kernel_initializer=init, \
                input_shape=input_shape  \
              )
    else:
       return Dense(
                filters,                 \
                kernel_initializer=init  \
              )


# deconv 層のファクトリーメソッド
def deconvLayer(filters, init="he_normal", input_shape=None):
    if input_shape is not None:
        return Conv2DTranspose(          \
                filters,                 \
                (5, 5),                  \
                strides=(2, 2),          \
                kernel_initializer=init, \
                padding="same",          \
                input_shape=input_shape  \
               )
    else:
        return Conv2DTranspose(          \
                filters,                 \
                (5, 5),                  \
                strides=(2, 2),          \
                kernel_initializer=init, \
                padding="same"           \
               )


# conv 層のファクトリーメソッド
def convLayer(filters, init="he_normal", input_shape=None):
    if input_shape is not None:
        return Conv2D(                   \
                filters,                 \
                (5, 5),                  \
                strides=(2, 2),          \
                kernel_initializer=init, \
                input_shape=input_shape  \
               )
    else:
        return Conv2D(                   \
                filters,                 \
                (5, 5),                  \
                strides=(2, 2),          \
                kernel_initializer=init  \
               )


def generator_model():
    model = Sequential()
    layerSize   = int(IMG_SIZE/16)
    firstSize   = layerSize*layerSize*KERNEL_CORE_SIZE*8
    input_shape = (NOIZE_SIZE, )
    model.add(denseLayer(firstSize, input_shape=input_shape))
    model.add(Reshape((layerSize, layerSize, KERNEL_CORE_SIZE*8)))
    model.add(BatchNormalization(momentum=BN_M, epsilon=BN_E))
    model.add(Activation("relu"))
    model.add(deconvLayer(KERNEL_CORE_SIZE*4))
    model.add(BatchNormalization(momentum=BN_M, epsilon=BN_E))
    model.add(Activation("relu"))
    model.add(deconvLayer(KERNEL_CORE_SIZE*2))
    model.add(BatchNormalization(momentum=BN_M, epsilon=BN_E))
    model.add(Activation("relu"))
    model.add(deconvLayer(KERNEL_CORE_SIZE*1))
    model.add(BatchNormalization(momentum=BN_M, epsilon=BN_E))
    model.add(Activation("relu"))
    model.add(deconvLayer(3, init="glorot_normal"))
    model.add(Activation("tanh"))
    return model

def encoder_model():
    model = Sequential()
    input_shape = (IMG_SIZE, IMG_SIZE, 3)
    model.add(convLayer(KERNEL_CORE_SIZE*1, input_shape=input_shape))
    model.add(LeakyReLU(0.2))
    model.add(convLayer(KERNEL_CORE_SIZE*2, input_shape=input_shape))
    model.add(BatchNormalization(momentum=BN_M, epsilon=BN_E))
    model.add(LeakyReLU(0.2))
    model.add(convLayer(KERNEL_CORE_SIZE*4, input_shape=input_shape))
    model.add(BatchNormalization(momentum=BN_M, epsilon=BN_E))
    model.add(LeakyReLU(0.2))
    model.add(convLayer(KERNEL_CORE_SIZE*8, input_shape=input_shape))
    model.add(BatchNormalization(momentum=BN_M, epsilon=BN_E))
    model.add(LeakyReLU(0.2))
    model.add(Flatten())
    model.add(denseLayer(NOIZE_SIZE, init="glorot_normal"))
    model.add(Activation("tanh"))
    return model

# 画像を出力する
# 学習画像と出力画像を引数に，左右に並べて一枚の画像として出力
# 学習画像と出力画像は同じサイズ，枚数を前提
def combine_images(learn, epoch, batch, path="output/"):
    total  = learn[0].shape[0]
    cols   = int(math.sqrt(total))
    rows   = math.ceil(float(total)/cols)
    w, h   = learn[0].shape[1:3]
    size   = (h*rows*2, w*cols*2, 3)
    output = np.zeros(size, dtype=learn[0].dtype)

    for n in range(len(learn[0])):
        i = int(n/cols)
        j = n % cols
        w0 = w* i
        w1 = w*(i+1)
        w2 = w*(cols+i)
        w3 = w*(cols+i+1)
        h0 = h* j
        h1 = h*(j+1)
        h2 = h*(rows+j)
        h3 = h*(rows+j+1)
        for k in range(3):
            output[w0:w1, h0:h1, k] = learn[0][n][:, :, k]
            output[w0:w1, h2:h3, k] = learn[1][n][:, :, k]
            output[w2:w3, h2:h3, k] = learn[2][n][:, :, k]
            output[w2:w3, h0:h1, k] = learn[3][n][:, :, k]

    output = output*127.5 + 127.5
    if not os.path.exists(GENERATED_IMAGE_PATH):
        os.mkdir(GENERATED_IMAGE_PATH)
    imgPath  = GENERATED_IMAGE_PATH
    imgPath += path
    imgPath += "%04d_%04d.png" % (epoch, batch)
    cv2.imwrite(imgPath, output.astype(np.uint8))

    return output

def train():
    (datas, _), (_, _) = FriendsLoader.load_data()
    datas = (datas.astype(np.float32) - 127.5)/127.5
    shape = datas.shape
    datas = datas.reshape(shape[0], shape[1], shape[2], 3)

    g_json_path    = SAVE_MODEL_PATH + "generator.json"
    g_weights_path = SAVE_MODEL_PATH + "g_w_"
    g_opt          = Adam(lr=G_LR, beta_1=G_BETA)
    e_json_path    = SAVE_MODEL_PATH + "encoder.json"
    e_weights_path = SAVE_MODEL_PATH + "e_w_"
    e_opt          = Adam(lr=E_LR, beta_1=E_BETA)
  
    if os.path.exists(SAVE_MODEL_PATH) == False:
        os.mkdir(SAVE_MODEL_PATH)
    if os.path.exists(SAVE_NOIZE_PATH) == False:
        os.mkdir(SAVE_NOIZE_PATH)
 
    # Encoder のロード
    if os.path.exists(e_json_path):
        with open(e_json_path, "r", encoding="utf-8") as f:
            encoder = model_from_json(f.read())
    else:
        encoder = encoder_model()
    e_weights_load_path = e_weights_path + str(START_EPOCH) + ".h5"
    if os.path.exists(e_weights_load_path):
        encoder.load_weights(e_weights_load_path, by_name=False)
    with open(e_json_path, "w", encoding="utf-8") as f:
        f.write(encoder.to_json())
    encoder.summary()

    # Generator のロード
    if os.path.exists(g_json_path):
        with open(g_json_path, "r", encoding="utf-8") as f:
            generator = model_from_json(f.read())
    else:
        generator = generator_model()
    g_weights_load_path = g_weights_path + str(START_EPOCH) + ".h5"
    if os.path.exists(g_weights_load_path):
        generator.load_weights(g_weights_load_path, by_name=False)
    with open(g_json_path, "w", encoding="utf-8") as f:
        f.write(generator.to_json())
    generator.summary()

    # コンパイル
    initializer = Sequential([encoder, generator])
    initializer.compile(loss="mean_squared_error", \
                            optimizer=e_opt, metrics=["accuracy"])

    num_batches = int(datas.shape[0] / BATCH_SIZE)
    print('Number of batches:', num_batches)
    
    # ログを出力する
    logfile = open("tmp/logdata.txt", "w", encoding="utf-8")

    # AutoEncoder を再帰的に学習する
    # r は再帰回数
    r = 4
    for epoch in range(NUM_EPOCH):
        X = list(datas)
        y = list(datas)
        for i in range(r):
            X += list(initializer.predict(np.array(X[-1*datas.shape[0]:]), verbose=0))
            y += list(datas)
        i_loss = initializer.fit(np.array(X), np.array(y), \
                batch_size=BATCH_SIZE, epochs=1)
        loss = []
        for i in range(r):
            t = [i_loss.history["loss"][i*BATCH_SIZE:(i+1)*BATCH_SIZE]]
            t += [i_loss.history["acc"][i*BATCH_SIZE:(i+1)*BATCH_SIZE]]
            loss.append(t)

        # 出力の様子を確認
        t   = "epoch: %d"
        tp  = [epoch]
        for i in range(r):
            t  += "loss_" + str(i) + ": [%f, %f], "
            tp += loss[i]
        print(t % tuple(tp))
        logfile.write((t+"\n") % tuple(tp))

        # 生成画像を出力
        l = []
        for i in range(4):
            l.append([output[i][20*j] for j in range(BATCH_SIZE)])
        combine_images(l, epoch, 0)

if __name__ == "__main__":
    train()
