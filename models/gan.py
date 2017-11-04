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
from setting import BATCH_SIZE
from setting import NUM_EPOCH
from setting import SPAN_UPDATE_NOIZE

from setting import G_LR
from setting import G_BETA
from setting import D_LR
from setting import D_BETA

SAVE_MODEL_PATH = "tmp/save_models/"
SAVE_NOIZE_PATH = "tmp/save_noizes/"
GENERATED_IMAGE_PATH = "tmp/"


def generator_model():
    layerSize = int(IMG_SIZE/16)
    model = Sequential()
    model.add(Dense(layerSize*layerSize*512, input_shape=(100,)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Reshape((layerSize, layerSize, 512)))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(256, (5, 5), padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(128, (5, 5), padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D( 64, (5, 5), padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(3, (5, 5), padding="same"))
    model.add(Activation("tanh"))
    return model

def discriminator_model():
    model = Sequential()
    model.add(Conv2D(64, (5, 5), strides=(2, 2),
                    input_shape=(IMG_SIZE, IMG_SIZE, 3))) # ここ注意
    model.add(LeakyReLU(0.2))
    model.add(Conv2D(128, (5, 5), strides=(2, 2)))
    model.add(LeakyReLU(0.2))
    model.add(Conv2D(256, (5, 5), strides=(2, 2)))
    model.add(LeakyReLU(0.2))
    model.add(Conv2D(512, (5, 5), strides=(2, 2)))
    model.add(LeakyReLU(0.2))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(100))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation("sigmoid"))
    return model

# 画像を出力する
# 学習画像と出力画像を引数に，左右に並べて一枚の画像として出力
# 学習画像と出力画像は同じサイズ，枚数を前提
def combine_images(learned, generated, epoch, batch, path=""):
    total = generated.shape[0]
    cols = int(math.sqrt(total))
    rows = math.ceil(float(total)/cols)
    width, height = generated.shape[1:3]
    combined_image = np.zeros((height*rows, width*cols*2, 3),dtype=generated.dtype)

    for index in range(len(generated)):
        l = learned[index]
        g = generated[index]
        i = int(index/cols)
        j = index % cols
        for k in range(3):
            combined_image[width*i:width*(i+1), height*j:height*(j+1), k] = l[:, :, k]
            combined_image[width*i:width*(i+1), height*(rows+j):height*(rows+j+1), k] = g[:, :, k]

    combined_image = combined_image*127.5 + 127.5
    if not os.path.exists(GENERATED_IMAGE_PATH):
        os.mkdir(GENERATED_IMAGE_PATH)
    imgPath  = GENERATED_IMAGE_PATH
    imgPath += path
    imgPath += "%04d_%04d.png" % (epoch, batch)
    cv2.imwrite(imgPath, combined_image.astype(np.uint8))

    return combined_image

def train():
    (X_train, y_train), (_, _) = FriendsLoader.load_data()
    X_train = (X_train.astype(np.float32) - 127.5)/127.5
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 3)

    d_json_path    = SAVE_MODEL_PATH + "discriminator.json"
    d_weights_path = SAVE_MODEL_PATH + "discriminator.h5"
    d_opt          = Adam(lr=D_LR, beta_1=D_BETA)
    g_json_path    = SAVE_MODEL_PATH + "generator.json"
    g_weights_path = SAVE_MODEL_PATH + "generator.h5"
    g_opt          = Adam(lr=G_LR, beta_1=G_BETA)
    
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

    # generator+discriminator （discriminator部分の重みは固定）
    discriminator.trainable = False
    if os.path.exists(g_json_path):
        with open(g_json_path, "r", encoding="utf-8") as f:
            generator = model_from_json(f.read())
    else:
        generator = generator_model()
    if os.path.exists(g_weights_path):
        generator.load_weights(g_weights_path, by_name=False)
    dcgan = Sequential([generator, discriminator])
    dcgan.compile(loss="binary_crossentropy", optimizer=g_opt)
    with open(g_json_path, "w", encoding="utf-8") as f:
        f.write(generator.to_json())

    num_batches = int(X_train.shape[0] / BATCH_SIZE)
    print("Number of batches:", num_batches)

    # 出力画像用のノイズを生成
    # 画像の成長過程を見たいので，出力画像には常に同じノイズを使う
    n_sample_path = SAVE_NOIZE_PATH + "forSample.dill"

    if os.path.exists(n_sample_path):
        with open(n_sample_path, "rb") as f:
            n_sample = dill.load(f)
    else:
        n_sample = np.array([np.random.uniform(-1, 1, 100) for _ in range(BATCH_SIZE)])
        with open(n_sample_path, "wb") as f:
            dill.dump(n_sample, f)

    for epoch in range(NUM_EPOCH):
        # 学習に使用するノイズを取得
        # 同じノイズを使い続ける方が学習速度は速いが汎化性能が低い
        # 試しに 100 エポックごとにノイズを変えてみる
        if epoch % SPAN_UPDATE_NOIZE == 0:
            n_learn = np.array([np.random.uniform(-1, 1, 100) for _ in range(BATCH_SIZE)])

        for index in range(num_batches):
            image_batch = X_train[index*BATCH_SIZE:(index+1)*BATCH_SIZE]
            generated_images = generator.predict(n_learn, verbose=0)

            # discriminatorを更新
            X = np.concatenate((image_batch, generated_images))
            y = [1]*BATCH_SIZE + [0]*BATCH_SIZE
            d_loss = discriminator.train_on_batch(X, y)

            # generatorを更新
            g_loss = dcgan.train_on_batch(n_learn, [1]*BATCH_SIZE)
            print("epoch: %d, batch: %d, g_loss: %f, d_loss: %f" % (epoch, index, g_loss, d_loss))

            # 生成画像を出力
            if index % 700 == 0:
                sampled_images = generator.predict(n_sample, verbose=0)
                combine_images(generated_images, sampled_images, epoch, index, path="output/")
                # combine_images(generated_images, epoch, index, path="learned/")


        generator.save_weights(g_weights_path)
        discriminator.save_weights(d_weights_path)

if __name__ == "__main__":
    train()
