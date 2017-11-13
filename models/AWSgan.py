from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense, Activation, Reshape
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Flatten, Dropout
import math
import numpy as np
import os
from keras.datasets import mnist
from keras.optimizers import Adam
import models.FriendsLoader as FriendsLoader
import cv2
import dill

from setting import IMG_SIZE
from setting import NOIZE_SIZE
from setting import BATCH_SIZE
from setting import START_EPOCH
from setting import NUM_EPOCH
from setting import SPAN_UPDATE_NOIZE

from setting import NEXT_PATTERN

from setting import G_LR
from setting import G_BETA
from setting import D_LR
from setting import D_BETA

from models.InputManager import InputManager

SAVE_MODEL_PATH = "tmp/save_models/"
SAVE_NOIZE_PATH = "tmp/save_noizes/"
GENERATED_IMAGE_PATH = "tmp/"


def generator_model():
    layerSize = int(IMG_SIZE/16)
    model = Sequential()
    model.add(Dense(layerSize*layerSize*1024, input_shape=(NOIZE_SIZE,)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Reshape((layerSize, layerSize, 1024)))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(512, (5, 5), padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(256, (5, 5), padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(128, (5, 5), padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(3, (5, 5), padding="same"))
    model.add(Activation("tanh"))
    return model

def discriminator_model():
    model = Sequential()
    model.add(Conv2D(128, (5, 5), strides=(2, 2),
                    input_shape=(IMG_SIZE, IMG_SIZE, 3))) # ここ注意
    model.add(LeakyReLU(0.2))
    model.add(Conv2D(256, (5, 5), strides=(2, 2)))
    model.add(LeakyReLU(0.2))
    model.add(Conv2D(512, (5, 5), strides=(2, 2)))
    model.add(LeakyReLU(0.2))
    model.add(Conv2D(1024, (5, 5), strides=(2, 2)))
    model.add(LeakyReLU(0.2))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(NOIZE_SIZE))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation("sigmoid"))
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
    (Xg, _), (_, _) = FriendsLoader.load_data()
    Xg = (Xg.astype(np.float32) - 127.5)/127.5
    Xg = Xg.reshape(Xg.shape[0], Xg.shape[1], Xg.shape[2], 3)

    d_json_path    = SAVE_MODEL_PATH + "discriminator.json"
    d_weights_path = SAVE_MODEL_PATH + "discriminator.h5"
    d_opt          = Adam(lr=D_LR, beta_1=D_BETA)
    g_json_path    = SAVE_MODEL_PATH + "generator.json"
    g_weights_path = SAVE_MODEL_PATH + "generator.h5"
    g_opt          = Adam(lr=G_LR, beta_1=G_BETA)
  
    if os.path.exists(SAVE_MODEL_PATH) == False:
        os.mkdir(SAVE_MODEL_PATH)
    if os.path.exists(SAVE_NOIZE_PATH) == False:
        os.mkdir(SAVE_NOIZE_PATH)
 
    # Discriminator のロード 
    if os.path.exists(d_json_path):
        with open(d_json_path, "r", encoding="utf-8") as f:
            discriminator = model_from_json(f.read())
    else:
        discriminator = discriminator_model()
    if os.path.exists(d_weights_path):
        discriminator.load_weights(d_weights_path, by_name=False)
    discriminator.compile(loss="binary_crossentropy", optimizer=d_opt)
    with open(d_json_path, "w", encoding="utf-8") as f:
        f.write(discriminator.to_json())
    discriminator.summary()

    # Generator のロードと DCGAN のコンパイル
    if os.path.exists(g_json_path):
        with open(g_json_path, "r", encoding="utf-8") as f:
            generator = model_from_json(f.read())
    else:
        generator = generator_model()
    if os.path.exists(g_weights_path):
        generator.load_weights(g_weights_path, by_name=False)
    # generator+discriminator （discriminator部分の重みは固定）
    dcgan = Sequential([generator, discriminator])
    discriminator.trainable = False
    dcgan.compile(loss="binary_crossentropy", optimizer=g_opt)
    with open(g_json_path, "w", encoding="utf-8") as f:
        f.write(generator.to_json())
    dcgan.summary()

    num_batches = int(Xg.shape[0] / BATCH_SIZE)
    print('Number of batches:', num_batches)
    
    # ノイズ処理用のマネージャインスタンスを生成
    manager = InputManager(NEXT_PATTERN)
    gList = None

    # ログを出力する
    logfile = open("tmp/logdata.txt", "w", encoding="utf-8")

    for epoch in range(START_EPOCH, NUM_EPOCH):
        # 次に学習に使用するノイズセットを取得する
        n_learn = manager.next(epoch, gList)

        for index in range(num_batches):
            g_images = generator.predict(n_learn, verbose=0)
            d_images = Xg[index*BATCH_SIZE:(index+1)*BATCH_SIZE]

            # discriminatorを更新
            Xd = np.concatenate((d_images, g_images))
            yd = [1]*BATCH_SIZE + [0]*BATCH_SIZE
            d_loss = discriminator.train_on_batch(Xd, yd)

            # generatorを更新
            g_loss = dcgan.train_on_batch(n_learn, [1]*BATCH_SIZE)
            text   = "epoch: %d, batch: %d, g_loss: %f, d_loss: %f"
            print(text % (epoch, index, g_loss, d_loss))
            logfile.write((text+"\n") % (epoch, index, g_loss, d_loss))

            # discriminator の結果を出力してみる
            gList = discriminator.predict(g_images)
            """
            dList = discriminator.predict(d_images)
            Zg = [int(i[0]>0.5) for i in gList]
            Zd = [int(i[0]>0.5) for i in dList]
            print("g_res:" + str(sum(Zg)) + "/" + str(BATCH_SIZE))
            print("d_res:" + str(sum(Zd)) + "/" + str(BATCH_SIZE))
            """

            # 生成画像を出力
            if index % 700 == 0:
                l = []
                for n in manager.noizeList:
                    l.append(generator.predict(n, verbose=0))
                combine_images(l, epoch, index)

        generator.save_weights(g_weights_path)
        discriminator.save_weights(d_weights_path)

if __name__ == "__main__":
    train()
