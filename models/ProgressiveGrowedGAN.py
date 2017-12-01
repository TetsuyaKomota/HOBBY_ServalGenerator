# coding = utf-8
# 2018 年のあれ

from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Reshape
from keras.layers.core import Lambda
from keras.layers.convolutional import UpSampling2D
from keras.layers import Conv2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.pooling import AveragePooling2D

import numpy as np

import models.FriendsLoader as FriendsLoader

# G, D それぞれ，次の3層を追加するメソッド
# G の pixel-wise 正規化の正規化関数
# train
#   αの扱い
#   3層挿入のタイミング

# 3層追加メソッド
# filters : 入力の filter 数
def getAdditionalBlock_G(filters):
    model = Sequential()
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(filters/2, (3, 3), padding="same"))
    model.add(Lambda(lambda x:x/np.sum(x**2+1e-8, axis=0)))
    model.add(LeakyReLU(0.2))
    model.add(Conv2D(filters/2, (3, 3), padding="same"))
    model.add(Lambda(lambda x:x/np.sum(x**2+1e-8, axis=0)))
    model.add(LeakyReLU(0.2))
    return model

# filters : 出力の filter 数
def getAdditionalBlock_D(filters):
    model = Sequential()
    model.add(Conv2D(filters/2, (3, 3), padding="same"))
    model.add(LeakyReLU(0.2))
    model.add(Conv2D(filters,   (3, 3), padding="same"))
    model.add(LeakyReLU(0.2))
    model.add(AveragePooling2D((2, 2)))
    return model

# D の入力層を生成
def getInputBlock_D(filters):
    model = Sequential()
    layerSize = 2048/filters
    model.add(Dense(layerSize*layerSize*filters, input_shape=(128*128*3)))
    model.add(Reshape(layerSize, layerSize, filters))
    model.add(Conv2D(filters, (1, 1), padding="same"))
    return model

# 最初のモデルを生成
def firstModel_G():
    model = Sequential()
    model.add(Dense(4*4*512, input_shape=(512, )))
    model.add(Reshape((4, 4, 512)))
    model.add(Conv2D(512, (4, 4), padding="same"))
    model.add(Lambda(lambda x:x/np.sum(x**2+1e-8, axis=0)))
    model.add(LeakyReLU(0.2))
    model.add(Conv2D(512, (3, 3), padding="same"))
    model.add(Lambda(lambda x:x/np.sum(x**2+1e-8, axis=0)))
    model.add(LeakyReLU(0.2))
    return model

def firstModel_D():
    model = Sequential()
    model.add(Lambda(lambda x:K.concatinate(K.std(x, axis=0, keepims=True), x, axis=0), input_shape=(512, 4, 4)))
    model.add(Conv2D(512, (3, 3), padding="same"))
    model.add(LeakyReLU(0.2))
    model.add(Conv2D(512, (4, 4), padding="same"))
    model.add(LeakyReLU(0.2))
    model.add(Dense(1))
    return model

# 学習
def train():
    (datas, _), (_, _) = FriendsLoader.load_data()
    datas = (datas.astype(np.float32) - 127.5)/127.5
    shape = datas.shape
    datas = datas.reshape(shape[0], shape[1], shape[2], 3)

    # 各モデルをロード
    # メモリ的に 128*128 を最終目標
    # つまり 本体 + 5 ブロック (4 * 2**5 = 128)
    models_G = []
    models_D = []
    models_G.append(firstModel_G())
    models_D.append(firstModel_D())
    for i in range(5):
        models_G.append(getAdditionalBlock_G(512/(2**i)))
        models_D.append(getAdditionalBlock_D(512/(2**i)))

    # 各段階で入出力層を補ってコンパイル
    gan = Sequential([models_G[0], models_D[0]])
    gan.compile(loss="binary_crossentropy", \
                        optimizer=g_opt, metrics=["accuracy"])
    models_D[0].compile(loss="binary_crossentropy", \
                        optimizer=d_opt, metrics=["accuracy"])
    compiled_G = [gan]
    compiled_D = [models_D[0]]
    
    for i in range(5+1):
        # G の出力層
        out_G = Conv2D(3, (1, 1), padding="same")
        compiled_G.append(models_G[:i+1] + out_G + models_D[:i+1][::-1])
        compiled_G[-1].compile(loss="binary_crossentropy", \
                        optimizer=g_opt, metrics=["accuracy"])

        # D の入力層
        in_D  = [getInputBlock_D(512/(2**(i)))]
        compiled_D.append([in_D, models_D[:i+1][::-1]])
        compiled_D[-1].compile(loss="binary_crossentropy", \
                        optimizer=g_opt, metrics=["accuracy"])

    # モデルを表示
    compiled_G[-1].summary()
    compiled_D[-1].summary()










