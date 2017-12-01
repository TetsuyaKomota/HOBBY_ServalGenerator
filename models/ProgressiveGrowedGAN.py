# coding = utf-8
# 2018 年のあれ

from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Reshape
from keras.layers import Flatten
from keras.layers.core import Lambda
from keras.layers.convolutional import UpSampling2D
from keras.layers import Conv2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.pooling import AveragePooling2D
from keras.optimizers import Adam

import math
import numpy as np
import cv2
import os

import models.FriendsLoader as FriendsLoader
from p_setting import D_LR
from p_setting import D_BETA1
from p_setting import D_BETA2
from p_setting import G_LR
from p_setting import G_BETA1
from p_setting import G_BETA2
from p_setting import BATCH_SIZE
from p_setting import NOIZE_SIZE
SAVE_MODEL_PATH = "tmp/save_models/"
SAVE_NOIZE_PATH = "tmp/save_noizes/"
GENERATED_IMAGE_PATH = "tmp/"


# G, D それぞれ，次の3層を追加するメソッド
# G の pixel-wise 正規化の正規化関数
# train
#   αの扱い
#   3層挿入のタイミング

# 3層追加メソッド
def getAdditionalBlock_G(idx):
    filters   =  8 * 2**(5-idx)
    layerSize =  4 * 2**idx
    model = Sequential()
    model.add(UpSampling2D((2, 2), input_shape=(layerSize, layerSize, 2*filters)))
    model.add(Conv2D(filters, (3, 3), padding="same"))
    model.add(Lambda(lambda x:K.l2_normalize(x, axis=3)))
    model.add(LeakyReLU(0.2))
    model.add(Conv2D(filters, (3, 3), padding="same"))
    model.add(Lambda(lambda x:K.l2_normalize(x, axis=3)))
    model.add(LeakyReLU(0.2))
    return model

def getAdditionalBlock_D(idx):
    filters   =  8 * 2**(5-idx)
    layerSize =  8 * 2**idx
    model = Sequential()
    model.add(Conv2D(  filters, (3, 3), padding="same", input_shape=(layerSize, layerSize, filters)))
    model.add(LeakyReLU(0.2))
    model.add(Conv2D(2*filters, (3, 3), padding="same"))
    model.add(LeakyReLU(0.2))
    model.add(AveragePooling2D((2, 2)))
    return model

# G の出力層を生成
def getOutputBlock_G(idx):
    model = Sequential()
    filters   = 16 * 2**(5-idx)
    layerSize =  4 * 2**idx
    model.add(Conv2D(3, (1, 1), padding="same", input_shape=(layerSize, layerSize, filters)))
    return model

# D の入力層を生成
def getInputBlock_D(idx):
    filters   = 16 * 2**(5-idx)
    layerSize =  4 * 2**idx
    model = Sequential()
    model.add(Conv2D(filters, (1, 1), padding="same", input_shape=(layerSize, layerSize, 3)))
    model.add(LeakyReLU(0.2))
    return model

# 最初のモデルを生成
def firstModel_G():
    model = Sequential()
    model.add(Dense(4*4*512, input_shape=(512, )))
    model.add(Reshape((4, 4, 512)))
    model.add(Conv2D(512, (4, 4), padding="same"))
    model.add(Lambda(lambda x:K.l2_normalize(x, axis=3)))
    model.add(LeakyReLU(0.2))
    model.add(Conv2D(512, (3, 3), padding="same"))
    model.add(Lambda(lambda x:K.l2_normalize(x, axis=3)))
    model.add(LeakyReLU(0.2))
    return model

def firstModel_D():
    model = Sequential()
    model.add(Lambda(lambda x:K.concatenate([K.std(x, axis=3, keepdims=True), x], axis=3), input_shape=(4, 4, 512)))
    model.add(Conv2D(512, (3, 3), padding="same"))
    model.add(LeakyReLU(0.2))
    model.add(Conv2D(512, (4, 4), padding="same"))
    model.add(LeakyReLU(0.2))
    model.add(Flatten())
    model.add(Dense(1))
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

# 学習
def train():
    (originals, _), (_, _) = FriendsLoader.load_data()
    datas = (originals.astype(np.float32) - 127.5)/127.5
    shape = datas.shape
    datas = datas.reshape(shape[0], shape[1], shape[2], 3)

    g_opt = Adam(lr=G_LR, beta_1=G_BETA1, beta_2=G_BETA2)
    d_opt = Adam(lr=D_LR, beta_1=D_BETA1, beta_2=D_BETA2)

    # 各モデルをロード
    # メモリ的に 128*128 を最終目標
    # つまり 本体 + 5 ブロック (4 * 2**5 = 128)
    models_G   = []
    models_D   = []
    models_G_O = []
    models_D_I = []
    models_G.append(firstModel_G())
    models_D.append(firstModel_D())
    models_G_O.append(getOutputBlock_G(0))
    models_D_I.append(getInputBlock_D(0))
    for i in range(5):
        models_G.append(getAdditionalBlock_G(i))
        models_D.append(getAdditionalBlock_D(i))
        models_G_O.append(getOutputBlock_G(i+1))
        models_D_I.append(getInputBlock_D(i+1))

    # 各段階で入出力層を補ってコンパイル
    compiled_G = []
    compiled_D = []
    for i in range(5+1):
        print(i)
        # G の出力層
        out_G = models_G_O[i]
        # D の入力層
        in_D  = models_D_I[i]

        compiled_D.append(Sequential([in_D] + models_D[:i+1][::-1]))
        compiled_G.append(Sequential(models_G[:i+1] + [out_G, in_D] + models_D[:i+1][::-1]))

        compiled_G[-1].compile(loss="binary_crossentropy", \
                        optimizer=g_opt, metrics=["accuracy"])

        compiled_D[-1].compile(loss="binary_crossentropy", \
                        optimizer=d_opt, metrics=["accuracy"])

        # とりあえず表示
        compiled_G[-1].summary()
        compiled_D[-1].summary()


    num_batches = int(datas.shape[0] / BATCH_SIZE)
    print('Number of batches:', num_batches)

    # ログを出力する
    logfile = open("tmp/logdata.txt", "w", encoding="utf-8")
    
    # 小さいモデルから学習する
    for i in range(5+1):
        # 画像生成用の G をコンパイル
        generator = Sequential(models_G[:i+1] + [models_G_O[i]])
        generator.trainable = False
        generator.compile(loss="binary_crossentropy", optimizer=g_opt)
        # モデルに合わせてリアルデータを縮小する
        resized = []   
        for d in originals:
            resized.append(cv2.resize(d, (4*2**i, 4*2**i), interpolation=cv2.INTER_LINEAR))
        resized = np.array(resized)
        datas = (resized.astype(np.float32) - 127.5)/127.5
        shape = datas.shape
        datas = datas.reshape(shape[0], shape[1], shape[2], 3)

        for epoch in range(60):
            for index in range(num_batches):
                noize = np.array([np.random.uniform(-1,1,NOIZE_SIZE) for _ in range(BATCH_SIZE)])
                
                g_images = generator.predict(noize, verbose=0)
                d_images = datas[index*BATCH_SIZE:(index+1)*BATCH_SIZE]

                # D を更新
                Xd = np.concatenate((d_images, g_images))
                yd = [1]*BATCH_SIZE + [0]*BATCH_SIZE
                d_loss = compiled_D[i].fit(Xd, yd, shuffle=False, epochs=1, batch_size=BATCH_SIZE, verbose=0)
                d_loss = [d_loss.history["loss"][-1],d_loss.history["acc"][-1]]

                # G を更新
                Xg = noize
                yg = [1]*BATCH_SIZE
                g_loss = compiled_G[i].fit(Xg, yg, shuffle=False, epochs=2, batch_size=BATCH_SIZE, verbose=0)
                g_loss = [g_loss.history["loss"][-1],g_loss.history["acc"][-1]]

                # D の出力の様子を確認
                t   = "epoch: %d, batch: %d, "
                t  += "g_loss: [%f, %f], d_loss: [%f, %f], "
                tp  = [epoch, index]
                tp += g_loss
                tp += d_loss
                print(t % tuple(tp))
                logfile.write((t+"\n") % tuple(tp))

                # 生成画像を出力
                if index % int(num_batches/2) == 0:
                    l = []
                    l.append(d_images)
                    l.append(generator.predict(noize, verbose=0))
                    l.append(generator.predict(noize, verbose=0))
                    l.append(generator.predict(noize, verbose=0))
                    combine_images(l, epoch, index)


        exit()

