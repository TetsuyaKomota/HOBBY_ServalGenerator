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

def discriminator_model():
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
    model.add(denseLayer(1, init="glorot_normal"))
    model.add(Activation("sigmoid"))
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

    d_json_path    = SAVE_MODEL_PATH + "discriminator.json"
    d_weights_path = SAVE_MODEL_PATH + "d_w_"
    d_opt          = Adam(lr=D_LR, beta_1=D_BETA)
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
 
    # Discriminator のロード 
    if os.path.exists(d_json_path):
        with open(d_json_path, "r", encoding="utf-8") as f:
            discriminator = model_from_json(f.read())
    else:
        discriminator = discriminator_model()
    d_weights_load_path = d_weights_path + str(START_EPOCH) + ".h5"
    if os.path.exists(d_weights_load_path):
        discriminator.load_weights(d_weights_load_path, by_name=False)
    discriminator.compile(loss="binary_crossentropy", \
                            optimizer=d_opt, metrics=["accuracy"])
    with open(d_json_path, "w", encoding="utf-8") as f:
        f.write(discriminator.to_json())
    discriminator.summary()

    # Generator のロードと DCGAN のコンパイル
    if os.path.exists(g_json_path):
        with open(g_json_path, "r", encoding="utf-8") as f:
            generator = model_from_json(f.read())
    else:
        generator = generator_model()
    g_weights_load_path = g_weights_path + str(START_EPOCH) + ".h5"
    if os.path.exists(g_weights_load_path):
        generator.load_weights(g_weights_load_path, by_name=False)
    # generator+discriminator （discriminator部分の重みは固定）
    dcgan = Sequential([generator, discriminator])
    discriminator.trainable = False
    dcgan.compile(loss="binary_crossentropy", \
                            optimizer=g_opt, metrics=["accuracy"])
    with open(g_json_path, "w", encoding="utf-8") as f:
        f.write(generator.to_json())
    dcgan.summary()

    # Encoder のロードと AutoEncoder のコンパイル
    if os.path.exists(e_json_path):
        with open(e_json_path, "r", encoding="utf-8") as f:
            encoder = model_from_json(f.read())
    else:
        encoder = encoder_model()
    e_weights_load_path = e_weights_path + str(START_EPOCH) + ".h5"
    if os.path.exists(e_weights_load_path):
        encoder.load_weights(e_weights_load_path, by_name=False)
    # encoder+generator
    # initializer は G を固定せず，G の初期化を担当
    # autoencoder は G を固定して，E を学習する
    initializer = Sequential([encoder, generator])
    autoencoder = Sequential([encoder, generator])
    initializer.compile(loss="mean_squared_error", \
                            optimizer=e_opt, metrics=["accuracy"])
    generator.trainable = False
    autoencoder.compile(loss="mean_squared_error", \
                            optimizer=e_opt, metrics=["accuracy"])
    with open(e_json_path, "w", encoding="utf-8") as f:
        f.write(encoder.to_json())
    autoencoder.summary()

    num_batches = int(datas.shape[0] / BATCH_SIZE)
    print('Number of batches:', num_batches)
    
    # ノイズ処理用のマネージャインスタンスを生成
    manager = InputManager(NEXT_PATTERN)
    gList = None

    # ログを出力する
    logfile = open("tmp/logdata.txt", "w", encoding="utf-8")

    for epoch in range(START_EPOCH, NUM_EPOCH):
        np.random.shuffle(datas)
       
        # エポックごとに，G をエンコーダ目線で初期化する
        if epoch % 1 == 0:
            i_loss = initializer.fit(datas, datas, epochs=1)
            i_loss = [i_loss.history["loss"][-1],i_loss.history["acc"][-1]]
 
        # 学習した Encoder で，real のノイズ値を生成する
        n_encode = encoder.predict(datas, verbose=0)

        for index in range(num_batches):
            # 次に学習に使用するノイズセットを取得する
            batch_g = int(BATCH_SIZE/2)
            batch_e = BATCH_SIZE - batch_g
            n_learn = manager.next(epoch, gList)
            np.random.shuffle(n_encode)
            g_images = generator.predict(n_learn[:batch_g], verbose=0)
            e_images = generator.predict(n_encode[:batch_e], verbose=0)
            d_images = datas[index*BATCH_SIZE:(index+1)*BATCH_SIZE]

            # discriminatorを更新
            Xd = np.concatenate((d_images, g_images, e_images))
            yd = [1]*BATCH_SIZE + [0]*BATCH_SIZE
            d_batch_size = BATCH_SIZE *(1 + int(epoch>=600))
            d_loss = discriminator.fit(Xd, yd, batch_size=d_batch_size, \
             epochs=epoch+1, shuffle=False, verbose=0, initial_epoch=epoch)
            d_loss = [d_loss.history["loss"][-1],d_loss.history["acc"][-1]]

            # generatorを更新
            # 論文通り，G の学習は 2 エポック行う
            Xg = n_learn
            yg = [1]*BATCH_SIZE
            g_loss = dcgan.fit(Xg, yg, batch_size=BATCH_SIZE, \
             epochs=epoch+2, shuffle=False, verbose=0, initial_epoch=epoch)
            g_loss = [g_loss.history["loss"][-1],g_loss.history["acc"][-1]]

            # encoderを更新
            Xe = d_images
            ye = d_images
            e_loss = autoencoder.fit(Xe, ye, batch_size=BATCH_SIZE, \
             epochs=epoch+2, shuffle=False, verbose=0, initial_epoch=epoch)
            e_loss = [e_loss.history["loss"][-1],e_loss.history["acc"][-1]]

            # 評価
            acc = discriminator.evaluate(Xd, yd, \
                                          batch_size=BATCH_SIZE, verbose=0)

            # D の出力の様子を確認
            pred     = discriminator.predict(Xd, verbose=0)
            pred_d   = pred[:BATCH_SIZE]
            pred_g   = pred[BATCH_SIZE:]
            pred_d_m = sum(pred_d)/BATCH_SIZE
            pred_g_m = sum(pred_g)/BATCH_SIZE
            pred_d_v = sum([(p-pred_d_m)**2 for p in pred_d])/BATCH_SIZE
            pred_g_v = sum([(p-pred_g_m)**2 for p in pred_g])/BATCH_SIZE
 
            t   = "epoch: %d, batch: %d, "
            t  += "g_loss: [%f, %f], d_loss: [%f, %f], "
            t  += "e_loss: [%f, %f], i_loss: [%f, %f], acc: [%f, %f], "
            t  += "predict(m, v)  g(%f, %f) d(%f, %f), "
            tp  = [epoch, index]
            tp += g_loss
            tp += d_loss
            tp += e_loss
            tp += i_loss
            tp += acc
            tp += [pred_g_m, pred_g_v, pred_d_m, pred_d_v]
            print(t % tuple(tp))
            logfile.write((t+"\n") % tuple(tp))

            # 生成画像を出力
            if index % int(num_batches/2) == 0 or index == num_batches-1:
                l = []
                l.append(generator.predict(manager.noizeList[0], verbose=0))
                l.append(generator.predict(n_encode[:BATCH_SIZE], verbose=0))
                l.append(generator.predict(manager.noizeList[1], verbose=0))
                l.append(generator.predict(manager.noizeList[2], verbose=0))
                combine_images(l, epoch, index)

        if epoch % 25 == 0:
            generator.save_weights(g_weights_path + str(epoch) + ".h5")
            discriminator.save_weights(d_weights_path + str(epoch) + ".h5")
            encoder.save_weights(e_weights_path + str(epoch) + ".h5")

if __name__ == "__main__":
    train()
