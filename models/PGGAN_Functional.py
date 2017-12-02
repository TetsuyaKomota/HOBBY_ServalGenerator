# coding = utf-8
# Sequential モデルに限界を感じたので functionalAPI で書き直してみる

from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense, Activation, Reshape
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

class LayerSet:

    def __init__(self):
        self.G   = self.firstModel_G()
        self.D   = self.firstModel_D()
        self.G_O = []
        self.G_A = []
        self.D_I = []
        self.D_A = []
        for i in range(5):
            self.G_A.append(self.getAdditionalBlock_G(i))
            self.D_A.append(self.getAdditionalBlock_D(i))
            self.G_O.append(self.getOutputBlock_G(i)) 
            self.D_I.append(self.getInputBlock_D(i))
        self.G_O.append(self.getOutputBlock_G(5+1)) 
        self.D_I.append(self.getInputBlock_D(5+1))

    # 3層追加メソッド
    def getAdditionalBlock_G(self, idx):
        filters   =  8 * 2**(5-idx)
        layerSize =  4 * 2**idx
        output = UpSampling2D((2, 2))
        output = Conv2D(filters, (3, 3), padding="same")(output)
        output = Lambda(lambda x:K.l2_normalize(x, axis=3))(output)
        output = LeakyReLU(0.2)(output)
        output = Conv2D(filters, (3, 3), padding="same")(output)
        output = Lambda(lambda x:K.l2_normalize(x, axis=3))(output)
        output = LeakyReLU(0.2)(output)
        return output

    def getAdditionalBlock_D(self, idx):
        filters   =  8 * 2**(5-idx)
        layerSize =  8 * 2**idx
        output = Conv2D(filters, (3, 3), padding="same")
        output = LeakyReLU(0.2)(output)
        output = Conv2D(2*filters, (3, 3), padding="same")(output)
        output = LeakyReLU(0.2)(output)
        output = AveragePooling2D((2, 2))(output)
        return output

    # G の出力層を生成
    def getOutputBlock_G(self, idx):
        filters   = 16 * 2**(5-idx)
        layerSize =  4 * 2**idx
        output = Conv2D(3, (1, 1), padding="same")
        output = Activation("tanh")(output)
        return output

    # D の入力層を生成
    def getInputBlock_D(self, idx):
        filters   = 16 * 2**(5-idx)
        layerSize =  4 * 2**idx
        output = Conv2D(filters, (1, 1), padding="same")
        output = LeakyReLU(0.2)(output)
        return output

    # 最初のモデルを生成
    def firstModel_G(self):
        output = Dense(4*4*512)
        output = Reshape((4, 4, 512))(output)
        output = Conv2D(512, (4, 4), padding="same")(output)
        output = Lambda(lambda x:K.l2_normalize(x, axis=3))(output)
        output = LeakyReLU(0.2)(output)
        output = Conv2D(512, (3, 3), padding="same")(output)
        output = Lambda(lambda x:K.l2_normalize(x, axis=3))(output)
        output = LeakyReLU(0.2)(output)
        return output

    def firstModel_D():
        output = Lambda(lambda x:K.concatenate([K.std(x, axis=3, keepdims=True), x], axis=3))
        output = Conv2D(512, (3, 3), padding="same")(output)
        output = LeakyReLU(0.2)(output)
        output = Conv2D(512, (4, 4), padding="same")(output)
        output = LeakyReLU(0.2)(output)
        output = Flatten()(output)
        output = Dense(1)(output)
        output = Activation("sigmoid")(output)
        return output

# 学習
def train():
    (originals, _), (_, _) = FriendsLoader.load_data()
    datas = (originals.astype(np.float32) - 127.5)/127.5
    shape = datas.shape
    datas = datas.reshape(shape[0], shape[1], shape[2], 3)

    g_opt = Adam(lr=G_LR, beta_1=G_BETA1, beta_2=G_BETA2)
    d_opt = Adam(lr=D_LR, beta_1=D_BETA1, beta_2=D_BETA2)

    # レイヤーセットをロード
    # メモリ的に 128*128 を最終目標
    # つまり 本体 + 5 ブロック (4 * 2**5 = 128)
    # モデルのコンパイルは必要に応じて適宜行う
    layerSet = LayerSet()

    num_batches = int(datas.shape[0] / BATCH_SIZE)
    print('Number of batches:', num_batches)

    # ログを出力する
    logfile = open("tmp/logdata.txt", "w", encoding="utf-8")
    
    # 小さいモデルから学習する
    for i in range(5+1):
        # 画像生成用の G をコンパイル
        input_G  = Input((512, ))
        output_G = layerSet.G(input_G)
        for j in range(i):
            output_G = layerSet.G_A[j](output_G)
        output_G = layerSet.G_O[i](output_G)
        generator = Model(inputs=input_G, outputs=output_G)
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
            # running Fade-in
            if i == 0:
                # 最初のモデルにはフェードイン必要なし
                break
            alpha = 0
            for epoch in range(60):
                alpha += 0.016

        # 学習モデルを構築
        input_D  = Input((4*2**i, 4*2**i, 3))
        output_D = layerSet.D_I[i](input_D)
        for j in range(i):
            output_D = layerSet.D_A[i-j-1](output_D)
        output_D = layerSet.D(output_D)
        discriminator = Model(inputs=input_D, outputs=output_D)
        discriminator.compile(loss="binary_crossentropy", \
                                    optimizer=d_opt, metrics=["accuracy"])

        input_G  = Input((512, ))
        output_G = layerSet.G(input_G)
        for j in range(i):
            output_G = layerSet.G_A[j](output_G)
        output_G = layerSet.G_O[i](output_G)
        output_G = layerSet.D_I[i](output_G, trainable=False)
        for j in range(i):
            output_G = layerSet.D_A[i-j-1](output_G, trainable=False)
        output_G = layerSet.D(output_G, trainable=False)
        gan = Model(inputs=input_G, outputs=output_G)
        gan.compile(loss="binary_crossentropy", \
                                    optimizer=g_opt, metrics=["accuracy"])
        
        for epoch in range(60):
            # finished Fade-in
            for index in range(num_batches):
                noize = np.array([np.random.uniform(-1,1,NOIZE_SIZE) for _ in range(BATCH_SIZE)])
                
                g_images = generator.predict(noize, verbose=0)
                d_images = datas[index*BATCH_SIZE:(index+1)*BATCH_SIZE]

                # D を更新
                Xd = np.concatenate((d_images, g_images))
                yd = [1]*BATCH_SIZE + [0]*BATCH_SIZE
                d_loss = discriminator.fit(Xd, yd, shuffle=False, epochs=1, batch_size=BATCH_SIZE, verbose=0)
                d_loss = [d_loss.history["loss"][-1],d_loss.history["acc"][-1]]

                # G を更新
                Xg = noize
                yg = [1]*BATCH_SIZE
                g_loss = gan.fit(Xg, yg, shuffle=False, epochs=2, batch_size=BATCH_SIZE, verbose=0)
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




