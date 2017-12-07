# coding = utf-8
# Sequential モデルに限界を感じたので functionalAPI で書き直してみる

from keras import backend as K
from keras.models import Sequential
from keras.models import Model
from keras.layers import Input, Dense, Activation, Reshape
from keras.layers import Flatten, Dropout
from keras.layers import Conv2D
from keras.layers.core import Lambda
from keras.layers.convolutional import UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.pooling import AveragePooling2D, MaxPooling2D
from keras.layers.merge import Add
from keras.optimizers import Adam
from keras.constraints import unit_norm, max_norm

import math
import numpy as np
import cv2
import os
import dill

import models.FriendsLoader as FriendsLoader
from p_setting import D_LR
from p_setting import D_BETA1
from p_setting import D_BETA2
from p_setting import G_LR
from p_setting import G_BETA1
from p_setting import G_BETA2
from p_setting import NUM_GENERATION
from p_setting import NUM_EPOCH
from p_setting import LOCAL_EPOCH
from p_setting import MAX_BATCH_SIZE
from p_setting import NOIZE_SIZE
from p_setting import START_GENERATION

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
        self.G_O.append(self.getOutputBlock_G(5)) 
        self.D_I.append(self.getInputBlock_D(5))
        self.load(START_GENERATION, 0)

    # レイヤーセットを引数に，trainable を変更する
    def setTrainable(self, layerList, isTrainable):
        for l in layerList:
            l.trainable = isTrainable

    # D の trainable を変更する
    # フェードインの際に再コンパイルをする必要があるため
    def setTrainableD(self, isTrainable):
        self.setTrainable(self.D, isTrainable)
        for d_i in self.D_I:
            self.setTrainable(d_i, isTrainable)
        for d_a in self.D_A:
            self.setTrainable(d_a, isTrainable)

    # 3層追加メソッド
    def getAdditionalBlock_G(self, idx):
        filters   =  2 * 2**(5-idx)
        output = []
        output.append(UpSampling2D((2, 2)))
        output.append(Conv2D(filters, (3, 3), padding="same", kernel_initializer="he_normal"))
        output.append(Lambda(lambda x:K.l2_normalize(x, axis=3)))
        output.append(LeakyReLU(0.2))
        output.append(Conv2D(filters, (3, 3), padding="same", kernel_initializer="he_normal"))
        output.append(Lambda(lambda x:K.l2_normalize(x, axis=3)))
        output.append(LeakyReLU(0.2))
        return output

    def getAdditionalBlock_D(self, idx):
        filters   =  2 * 2**(5-idx)
        output = []
        output.append(Conv2D(filters, (3, 3), padding="same", kernel_initializer="he_normal", kernel_constraint=max_norm()))
        output.append(LeakyReLU(0.2))
        output.append(Conv2D(2*filters, (3, 3), padding="same", kernel_initializer="he_normal", kernel_constraint=max_norm()))
        output.append(LeakyReLU(0.2))
        output.append(AveragePooling2D((2, 2)))
        output.append(Dropout(0.5))
        return output

    # G の出力層を生成
    def getOutputBlock_G(self, idx):
        output = []
        output.append(Conv2D(3, (1, 1), padding="same", kernel_initializer="he_normal"))
        output.append(Activation("tanh"))
        return output

    # D の入力層を生成
    def getInputBlock_D(self, idx):
        filters   = 4 * 2**(5-idx)
        output = []
        output.append(Conv2D(filters, (1, 1), padding="same", kernel_initializer="he_normal"))
        output.append(LeakyReLU(0.2))
        return output

    # 最初のモデルを生成
    def firstModel_G(self):
        output = []
        output.append(Dense(4*4*128, kernel_initializer="he_normal"))
        output.append(Reshape((4, 4, 128)))
        output.append(LeakyReLU(0.2))
        output.append(Conv2D(128, (4, 4), padding="same", kernel_initializer="he_normal"))
        output.append(LeakyReLU(0.2))
        output.append(Conv2D(128, (3, 3), padding="same", kernel_initializer="he_normal"))
        output.append(Lambda(lambda x:K.l2_normalize(x, axis=3)))
        output.append(LeakyReLU(0.2))
        return output

    def firstModel_D(self):
        output = []
        output.append(Lambda(lambda x:K.concatenate([K.std(x, axis=3, keepdims=True), x], axis=3)))
        output.append(Conv2D(128, (3, 3), padding="same", kernel_initializer="he_normal", kernel_constraint=max_norm()))
        output.append(LeakyReLU(0.2))
        output.append(Conv2D(128, (4, 4), padding="same", kernel_initializer="he_normal"))
        output.append(LeakyReLU(0.2))
        output.append(Flatten())
        output.append(Dropout(0.5))
        output.append(LeakyReLU(0.2))
        output.append(Dense(1))
        output.append(Activation("sigmoid"))
        return output

    # レイヤーの配列を組み立てる
    def build(self, layerList, inputs):
        output = inputs
        for l in layerList:
            output = l(output)
        return output

    # モデルをセーブする
    def save(self, generation, i):
        output = {}
        output["D"] = []
        for l in self.D:
            output["D"].append(l.get_weights())
        output["D_I"] = []
        for d_i in self.D_I:
            output["D_I"].append([])
            for l in d_i:
                output["D_I"][-1].append(l.get_weights())
        output["D_A"] = []
        for d_a in self.D_A:
            output["D_A"].append([])
            for l in d_a:
                output["D_A"][-1].append(l.get_weights())
        output["G"] = []
        for l in self.G:
            output["G"].append(l.get_weights())
        output["G_O"] = []
        for g_o in self.G_O:
            output["G_O"].append([])
            for l in g_o:
                output["G_O"][-1].append(l.get_weights())
        output["G_A"] = []
        for g_a in self.G_A:
            output["G_A"].append([])
            for l in g_a:
                output["G_A"][-1].append(l.get_weights())
        if os.path.exists("tmp/save_models/PGGAN/") == False:
            os.mkdir("tmp/save_models/PGGAN/")
        with open("tmp/save_models/PGGAN/weights_"+str(generation)+"_"+str(i)+".dill", "wb") as f:
            dill.dump(output, f)

    # モデルをロードする
    def load(self, generation, i):
        if os.path.exists("tmp/save_models/PGGAN/weights_"+str(generation)+"_"+str(i)+".dill") == False:
            print("[PGGAN_Functional]load:cannot load wieghts.dill")
            return 
        with open("tmp/save_models/PGGAN/weights_"+str(generation)+"_"+str(i)+".dill", "rb") as f:
            w = dill.load(f)
        for l in self.D:
            l.set_weights(w["D"].pop(0))
        for i, d_i in enumerate(self.D_I):
            for l in d_i:
                l.set_weights(w["D_I"][i].pop(0))
        for i, d_a in enumerate(self.D_A):
            for l in d_a:
                l.set_weights(w["D_A"][i].pop(0))
        for l in self.G:
            l.set_weights(w["G"].pop(0))
        for i, g_o in enumerate(self.G_O):
            for l in g_o:
                l.set_weights(w["G_O"][i].pop(0))
        for i, g_a in enumerate(self.G_A):
            for l in g_a:
                l.set_weights(w["G_A"][i].pop(0))

# 画像を出力する
# 学習画像と出力画像を引数に，左右に並べて一枚の画像として出力
# 学習画像と出力画像は同じサイズ，枚数を前提
def combine_images(learn, idx, fadefill, epoch, batch, path="output/"):
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
    imgPath += "%02d_%01d_%04d_%04d.png" % (idx, fadefill, epoch, batch)
    cv2.imwrite(imgPath, output.astype(np.uint8))

    return output

# 学習
def train():
    (originals, _), (_, _) = FriendsLoader.load_data()
    # 出力用のノイズを生成
    n_sample = np.array([np.random.uniform(-1,1,NOIZE_SIZE) for _ in range(2*MAX_BATCH_SIZE)])

    g_opt = Adam(lr=G_LR, beta_1=G_BETA1, beta_2=G_BETA2)
    d_opt = Adam(lr=D_LR, beta_1=D_BETA1, beta_2=D_BETA2)

    # レイヤーセットをロード
    # メモリ的に 128*128 を最終目標
    # つまり 本体 + 5 ブロック (4 * 2**5 = 128)
    # モデルのコンパイルは必要に応じて適宜行う
    l = LayerSet()

    # ログを出力する
    logfile = open("tmp/logdata.txt", "w", encoding="utf-8")
    
    # 小さいモデルから学習する
    for generation in range(NUM_GENERATION):
        for i in range(5+1):
            BATCH_SIZE = int(MAX_BATCH_SIZE / 4**i)
            num_batches = int(originals.shape[0] / BATCH_SIZE)
            print('Number of batches:', num_batches)
            # モデルに合わせてリアルデータを縮小する
            resized = []   
            for d in originals:
                resized.append(cv2.resize(d, (4*2**i, 4*2**i), interpolation=cv2.INTER_LINEAR))
            resized = np.array(resized)
            datas = (resized.astype(np.float32) - 127.5)/127.5
            shape = datas.shape
            datas = datas.reshape(shape[0], shape[1], shape[2], 3)

            # フェードを初期化する
            alpha = 0
            
            # フェードイン用のレイヤーを用意
            fade_D1 = Conv2D(4 * 2**(5-(i-1)), (1, 1), trainable=False)
            fade_D2 = Conv2D(4 * 2**(5-(i-1)), (1, 1), trainable=False)
            fade_G1 = Conv2D(               3, (1, 1), trainable=False)
            fade_G2 = Conv2D(               3, (1, 1), trainable=False)
            if i > 0:
                # running Fade-in
                # alpha を調節しながら学習する為，エポックごとにコンパイルする
                # 画像生成用の G をコンパイル
                # 学習済みのレイヤーは trainable=False
                l.setTrainable(l.G_O[i-1], False)
                input_G  = Input((128, ))
                output_G = l.build(l.G, input_G)
                for j in range(i-1):
                    # 学習済みのレイヤーは trainable=False
                    l.setTrainable(l.G_A[j], False)
                    output_G = l.build(l.G_A[j], output_G)
                output_G1 = l.build(l.G_O[i-1], output_G)
                output_G1 = UpSampling2D((2, 2))(output_G1)
                output_G1 = fade_G1(output_G1)
                output_G2 = l.build(l.G_A[i-1], output_G)
                output_G2 = l.build(l.G_O[i], output_G2)
                output_G2 = fade_G2(output_G2)
                output_G  = Add()([output_G1, output_G2])
                generator = Model(inputs=input_G, outputs=output_G)
                generator.compile(loss="binary_crossentropy", optimizer=g_opt)
                
                # 学習モデルを構築
                l.setTrainableD(True)
                # 学習済みのレイヤーは trainable=False
                l.setTrainable(l.D_I[i-1], False)
                input_D   = Input((4*2**i, 4*2**i, 3))
                output_D1 = AveragePooling2D((2, 2))(input_D)
                output_D1 = l.build(l.D_I[i-1], output_D1)
                output_D1 = fade_D1(output_D1)
                output_D2 = l.build(l.D_I[i], input_D)
                output_D2 = l.build(l.D_A[i-1], output_D2)
                output_D2 = fade_D2(output_D2)
                output_D  = Add()([output_D1, output_D2])
                for j in range(i-1):
                    # 学習済みのレイヤーは trainable=False
                    l.setTrainable(l.D_A[i-j-2], False)
                    output_D = l.build(l.D_A[i-j-2], output_D)
                output_D = l.build(l.D, output_D)
                discriminator = Model(inputs=input_D, outputs=output_D)
                discriminator.compile(loss="binary_crossentropy", \
                                    optimizer=d_opt, metrics=["accuracy"])
                
                l.setTrainableD(False)
                gan = Sequential([generator, discriminator])
                gan.compile(loss="binary_crossentropy", \
                                    optimizer=g_opt, metrics=["accuracy"])
            else:
                # 最初の学習モデルを構築
                # 画像生成用の G をコンパイル
                input_G  = Input((128, ))
                output_G = l.build(l.G, input_G)
                output_G = l.build(l.G_O[0], output_G)
                generator = Model(inputs=input_G, outputs=output_G)
                generator.compile(loss="binary_crossentropy", optimizer=g_opt)
     
                l.setTrainableD(True)
                input_D  = Input((4, 4, 3))
                output_D = l.build(l.D_I[0], input_D)
                output_D = l.build(l.D, output_D)
                discriminator = Model(inputs=input_D, outputs=output_D)
                discriminator.compile(loss="binary_crossentropy", \
                                    optimizer=d_opt, metrics=["accuracy"])

                l.setTrainableD(False)
                gan = Sequential([generator, discriminator])
                gan.compile(loss="binary_crossentropy", \
                                    optimizer=g_opt, metrics=["accuracy"])

            # とりあえず表示
            discriminator.summary() 
            gan.summary()

            # 各解像度の学習回数
            # 高解像度の時ほど学習回数を多くしている
            # 2倍にしているのは fade_in と full の分
            num_epoch = NUM_EPOCH * 2 * (i+1)
            for epoch in range(num_epoch):
                np.random.shuffle(datas)
                if len(fade_D1.get_weights()) > 0 and i > 0:
                    # fade レイヤーの重みを更新する
                    weights_size = 4 * 2**(5-(i-1))
                    # local_epoch/2 の時に alpha=1 に固定される
                    alpha  = min(alpha + 2.0/num_epoch, 1)
                    ALPHA1 = np.zeros((1, 1, weights_size, weights_size))
                    ALPHA2 = np.zeros((1, 1, weights_size, weights_size))
                    for k in range(ALPHA1.shape[2]):
                        ALPHA1[0, 0, k, k] = (1-alpha)
                        ALPHA2[0, 0, k, k] = alpha
                        # ALPHA1[0, 0, k, k] = 0
                        # ALPHA2[0, 0, k, k] = 1
                    fade_D1.set_weights([ALPHA1, fade_D1.get_weights()[1]])
                    fade_D2.set_weights([ALPHA2, fade_D2.get_weights()[1]])
                    ALPHA1 = np.zeros((1, 1, 3, 3))
                    ALPHA2 = np.zeros((1, 1, 3, 3))
                    for k in range(ALPHA1.shape[2]):
                        ALPHA1[0, 0, k, k] = (1-alpha)
                        ALPHA2[0, 0, k, k] = alpha
                        # ALPHA1[0, 0, k, k] = 0
                        # ALPHA2[0, 0, k, k] = 1

                    fade_G1.set_weights([ALPHA1, fade_G1.get_weights()[1]])
                    fade_G2.set_weights([ALPHA2, fade_G2.get_weights()[1]])

                for index in range(num_batches):
                    noize = np.array([np.random.uniform(-1,1,NOIZE_SIZE) for _ in range(BATCH_SIZE)])
                    
                    g_images = generator.predict(noize, verbose=0)
                    d_images = datas[index*BATCH_SIZE:(index+1)*BATCH_SIZE]

                    # D を更新
                    Xd = np.concatenate((d_images, g_images))
                    yd = [1]*BATCH_SIZE + [0]*BATCH_SIZE
                    d_loss = discriminator.fit(Xd, yd, shuffle=False, epochs=LOCAL_EPOCH, batch_size=BATCH_SIZE*2, verbose=0)
                    d_loss = [d_loss.history["loss"][-1],d_loss.history["acc"][-1]]

                    # G を更新
                    Xg = noize
                    yg = [1]*BATCH_SIZE
                    g_loss = gan.fit(Xg, yg, shuffle=False, epochs=LOCAL_EPOCH, batch_size=BATCH_SIZE, verbose=0)
                    g_loss = [g_loss.history["loss"][-1],g_loss.history["acc"][-1]]

                    # D の出力の様子を確認
                    t   = "full" if epoch >= NUM_EPOCH else "fade"
                    t  += "-I:%d, epoch: %d, batch: %d, "
                    t  += "g_loss: [%f, %f], d_loss: [%f, %f], "
                    tp  = [i, epoch, index]
                    tp += g_loss
                    tp += d_loss
                    print(t % tuple(tp))
                    logfile.write((t+"\n") % tuple(tp))

                    # 生成画像を出力
                    if index % int(num_batches/2 + 1) == 0:
                        fadefull = int(epoch >= NUM_EPOCH)
                        imgList = []
                        imgList.append(d_images)
                        imgList.append(generator.predict(noize, verbose=0))
                        imgList.append(generator.predict(n_sample[:BATCH_SIZE], verbose=0))
                        imgList.append(generator.predict(n_sample[MAX_BATCH_SIZE:MAX_BATCH_SIZE+BATCH_SIZE], verbose=0))
                        combine_images(imgList, i, fadefull, epoch, index)
        
            # 各解像度での学習終了時に重みを保存する
            l.save(generation, i) 

